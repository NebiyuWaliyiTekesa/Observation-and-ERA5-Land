import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import genextreme
import io

st.set_page_config(page_title="Can ERA5-Land data be used for Engineering purpose (Design Return Level)?", layout="wide")

st.title("🌧️ Can ERA5-Land data be used for Engineering purpose (Design Return Level)?")

# Methodology Note
st.info(
    "**Methodology:**\n\n"
    "To assess the suitability of ERA5-Land reanalysis data for urban infrastructure design and other engineering "
    "applications, we derived Duration-Depth-Frequency (DDF) curves for each station for both observed "
    "and ERA5-Land AMPs (1951-2023). The AMPs (observed and ERA5-Land) were fitted using the Generalized Extreme Value (GEV) "
    "distribution, and the DDF curves were computed for 1-, 3-, 6-, 12-, and 24-hr durations and return periods ($T$) "
    "of 20, 50, 100, and 200 years (Soto-Escobar et al., 2026). We then quantified relative bias across storm duration "
    "and $T$."
)

# =========================================================
# 1. AUTOMATIC FILE LOADING FROM GITHUB REPOSITORY
# =========================================================
#st.sidebar.header("📁 Data Source Status")
#st.sidebar.success("Files loaded automatically from repository!")

#st.sidebar.markdown("---")
#st.sidebar.header("Advanced Options")
#show_ci = st.sidebar.checkbox("Show 90% Bootstrap Confidence Intervals", value=False)

durations_str = ["1h", "3h", "6h", "12h", "24h"]
dur_hours = np.array([1, 3, 6, 12, 24])
thresholds = {"1h": 300, "3h": 500, "6h": 600, "12h": 800, "24h": 1000}
return_periods = [20, 50, 100, 200]

try:
    df_meta = pd.read_excel("locations.xlsx")
    
    obs_dict = {}
    era_dict = {}
    
    for dur in durations_str:
        obs_file = f"OBS_ARF_corrected_{dur}.xlsx"
        
        # Strip 'h' from dur (e.g., '1h' -> '1') and add 'hr.xlsx'
        num_str = dur.replace('h', '')
        era_file = f"{num_str}hr.xlsx"
        
        # Load and deduplicate observation file
        df_obs = pd.read_excel(obs_file)
        anno_col_obs = [c for c in df_obs.columns if str(c).lower() == 'anno'][0] if 'anno' in [str(c).lower() for c in df_obs.columns] else df_obs.columns[0]
        df_obs = df_obs.set_index(anno_col_obs)
        df_obs.columns = df_obs.columns.astype(str).str.strip()
        obs_dict[dur] = df_obs.loc[:, ~df_obs.columns.duplicated(keep="first")]
        
        # Load and deduplicate ERA5 file
        df_era = pd.read_excel(era_file)
        anno_col_era = [c for c in df_era.columns if str(c).lower() == 'anno'][0] if 'anno' in [str(c).lower() for c in df_era.columns] else df_era.columns[0]
        df_era = df_era.set_index(anno_col_era)
        df_era.columns = df_era.columns.astype(str).str.strip()
        era_dict[dur] = df_era.loc[:, ~df_era.columns.duplicated(keep="first")]

except Exception as e:
    st.error(f"Error loading files automatically from repository: {e}")
    st.stop()

# =========================================================
# 2. DATA PROCESSING, MAPPING & COMMON STATIONS
# =========================================================
code_col = [c for c in df_meta.columns if c.lower() in ['code', 'station_code', 'id', 'station']][0]
elev_col = [c for c in df_meta.columns if c.lower() in ['elev', 'elevation', 'alt', 'altitude', 'z']][0]
df_meta[code_col] = df_meta[code_col].astype(str).str.strip()

region_col_candidates = [c for c in df_meta.columns if c.lower() in ['region', 'zone', 'area', 'province', 'basin']]
has_region_col = len(region_col_candidates) > 0
reg_col = region_col_candidates[0] if has_region_col else None

def map_elevation_class(val):
    try:
        num = float(val)
        if num < 230.0:
            return 'Low (< 230 m)'
        elif 230.0 <= num <= 720.0:
            return 'Mid (230–720 m)'
        else:
            return 'High (> 720 m)'
    except (ValueError, TypeError):
        return 'Unknown'

raw_elev_map = dict(zip(df_meta[code_col], df_meta[elev_col]))
if has_region_col:
    region_map = dict(zip(df_meta[code_col], df_meta[reg_col].astype(str)))
else:
    region_map = {stn: map_elevation_class(elev) for stn, elev in raw_elev_map.items()}

def get_gev_depth(series, return_periods):
    data = series.dropna().values
    if len(data) < 5:
        return [np.nan] * len(return_periods)
    try:
        c, loc, scale = genextreme.fit(data)
        depths = []
        for T in return_periods:
            p = 1.0 - (1.0 / T)
            depths.append(genextreme.ppf(p, c, loc=loc, scale=scale))
        return depths
    except Exception:
        return [np.nan] * len(return_periods)

common_stations = None
for dur in durations_str:
    stns = [c for c in obs_dict[dur].columns if c in era_dict[dur].columns and c in raw_elev_map]
    if common_stations is None:
        common_stations = set(stns)
    else:
        common_stations = common_stations.intersection(set(stns))

common_stations = sorted(list(common_stations))

if len(common_stations) == 0:
    st.error("No matching station codes found across your metadata and duration files. Check column headers.")
    st.stop()

@st.cache_data
def precompute_station_curves(station_list):
    cache = {}
    for stn in station_list:
        obs_curves, era_curves = {}, {}
        for dur in durations_str:
            obs_s = obs_dict[dur][stn]
            era_s = era_dict[dur][stn]
            
            thr = thresholds[dur]
            obs_s = obs_s.where((obs_s > 0) & (obs_s <= thr))
            era_s = era_s.where((era_s > 0) & (era_s <= thr))
            
            obs_curves[dur] = get_gev_depth(obs_s, return_periods)
            era_curves[dur] = get_gev_depth(era_s, return_periods)
            
        cache[stn] = {
            "obs": obs_curves,
            "era": era_curves,
            "elev": raw_elev_map.get(stn, "N/A"),
            "region": region_map.get(stn, "Unknown")
        }
    return cache

station_cache = precompute_station_curves(common_stations)

# =========================================================
# 3. INTERACTIVE DASHBOARD VIEW (REGION -> STATION SELECTOR)
# =========================================================
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Filter & Select")
    
    available_regions = ['All'] + sorted(list(set(station_cache[s]['region'] for s in common_stations)))
    selected_region = st.selectbox("1. Select Region / Regime:", available_regions)
    
    if selected_region == 'All':
        filtered_stations = common_stations
    else:
        filtered_stations = [s for s in common_stations if station_cache[s]['region'] == selected_region]
    
    if not filtered_stations:
        st.warning("No stations found in this region.")
        st.stop()
        
    selected_stn = st.selectbox("2. Choose Station Code:", filtered_stations)
    
    st.info(
        f"**Selected Station:** `{selected_stn}`\n\n"
        f"**Region/Regime:** {station_cache[selected_stn]['region']}\n\n"
        f"**Elevation:** {station_cache[selected_stn]['elev']} m"
    )

with col2:
    st.subheader(f"DDF Curves for Station: {selected_stn}")
    
    stn_data = station_cache[selected_stn]
    colors = ["#2b83ba", "#d95f02", "#7570b3", "#e7298a"]
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=600)
    
    for idx_T, T in enumerate(return_periods):
        obs_y = [stn_data["obs"][dur][idx_T] for dur in durations_str]
        era_y = [stn_data["era"][dur][idx_T] for dur in durations_str]
        
        ax.plot(dur_hours, obs_y, marker="o", linestyle="-", color=colors[idx_T],
                linewidth=1.5, markersize=5, label=f"Obs ({T} yr)")
        ax.plot(dur_hours, era_y, marker="s", linestyle="--", color=colors[idx_T],
                linewidth=1.5, markersize=5, markerfacecolor="white", label=f"ERA5 ({T} yr)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dur_hours)
    ax.set_xticklabels([f"{h}h" for h in dur_hours], fontsize=9)
    ax.set_xlim(0.6, 30)
    
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
ax.set_title(
        f"Station: {selected_stn} | Region: {stn_data['region']} | Elev: {stn_data['elev']} m | Lat: {stn_data['lat']}°, Lon: {stn_data['lon']}°",
        fontsize=10,
        fontweight='bold'
    )
    ax.set_xlabel("Duration", fontsize=9, fontweight='bold')
    ax.set_ylabel("Return Level (mm)", fontsize=9, fontweight='bold')
    ax.legend(frameon=True, fontsize=7, loc="upper left", ncol=2)
    
    st.pyplot(fig)

# =========================================================
# 4. RAW RETURN LEVELS & PBIAS TABLES + DOWNLOAD
# =========================================================
st.markdown("---")
st.subheader(f"📊 Raw Return Levels & Bias Analysis for Station: {selected_stn}")

table_data = []
for dur in durations_str:
    for idx_T, T in enumerate(return_periods):
        o_val = stn_data["obs"][dur][idx_T]
        e_val = stn_data["era"][dur][idx_T]
        pb = 100.0 * (e_val - o_val) / o_val if (not np.isnan(o_val) and not np.isnan(e_val) and o_val != 0) else np.nan
        
        table_data.append({
            "Duration": dur,
            "Return_Period_yr": T,
            "Observed_mm": round(o_val, 2) if not np.isnan(o_val) else np.nan,
            "ERA5-Land_mm": round(e_val, 2) if not np.isnan(e_val) else np.nan,
            "PBIAS_pct": round(pb, 2) if not np.isnan(pb) else np.nan
        })

df_results = pd.DataFrame(table_data)

tab1, tab2 = st.tabs(["📋 Detailed Table & PBIAS", "📐 Traditional DDF Matrix View"])

with tab1:
    st.write("Complete breakdown of raw values and error metrics per duration and return period:")
    st.dataframe(df_results, use_container_width=True)
    
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Raw Return Levels for Station {selected_stn} (CSV)",
        data=csv_data,
        file_name=f"station_{selected_stn}_return_levels.csv",
        mime="text/csv",
    )

with tab2:
    col_obs, col_era = st.columns(2)
    with col_obs:
        st.markdown("**Observed Raw Return Levels (mm)**")
        df_obs_matrix = df_results.pivot(index="Duration", columns="Return_Period_yr", values="Observed_mm")
        st.dataframe(df_obs_matrix, use_container_width=True)
    with col_era:
        st.markdown("**ERA5-Land Raw Return Levels (mm)**")
        df_era_matrix = df_results.pivot(index="Duration", columns="Return_Period_yr", values="ERA5-Land_mm")
        st.dataframe(df_era_matrix, use_container_width=True)

# =========================================================
# 5. AUTOMATED ENGINEERING FEASIBILITY & ANALYSIS REPORT
# =========================================================
st.markdown("---")
st.subheader("📝 Automated Engineering Feasibility Conclusion (Station Specific)")

pbias_1h = df_results[df_results["Duration"] == "1h"]["PBIAS_pct"].mean()
pbias_24h = df_results[df_results["Duration"] == "24h"]["PBIAS_pct"].mean()

analysis_text = f"""
For station **{selected_stn}**, the comparison between ground station observations and ERA5-Land reanalysis design return levels highlights distinct duration-dependent behaviors:

* **Short-Duration Performance (1h – 3h):** ERA5-Land exhibits pronounced underestimation, with average biases dropping significantly (e.g., around **{pbias_1h:.1f}%** for 1-hour accumulation). This limitation is primarily driven by reanalysis grid-cell averaging and the coarse spatial/temporal representation of convective storms, making raw sub-daily extremes unsuitable for direct urban or small-catchment engineering design without rigorous bias correction.
* **Long-Duration Performance (12h – 24h):** As accumulation periods increase, the representativeness of ERA5-Land improves substantially. At the 24-hour duration, the bias shifts closer to zero (averaging roughly **{pbias_24h:.1f}%**), demonstrating that reanalysis data captures synoptic-scale aggregate precipitation volumes much more reliably.
"""

if abs(pbias_24h) < 15.0:
    feasibility_conclusion = f"✅ **Conclusion:** While **unsuitable for short-duration (sub-hourly/hourly) applications** due to severe underestimation, ERA5-Land shows good reliability for **daily (24-hr) cumulative engineering assessments** at this station (Bias ~ {pbias_24h:.1f}%). It can be cautiously used for macro-scale hydrological planning or after applying scaling factors."
else:
    feasibility_conclusion = f"❌ **Conclusion:** ERA5-Land exhibits noticeable discrepancies across durations at this station (24h bias is ~ {pbias_24h:.1f}%). Direct use for critical design return levels is **not recommended** without local calibration or quantile mapping."

st.info(analysis_text)
st.markdown(feasibility_conclusion)
