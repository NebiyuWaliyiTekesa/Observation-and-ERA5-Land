import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import genextreme
import io

st.set_page_config(page_title="Extreme Rainfall & DDF Viewer", layout="wide")

st.title("🌧️ Extreme Rainfall DDF Curve Viewer")
st.write("Automated comparison of Design Return Level derived from **Observed** ground station data and **ERA5-Land** reanalysis.")

# =========================================================
# 1. SIDEBAR FILE UPLOADERS
# =========================================================
st.sidebar.header("1. Upload Metadata File")
locations_file = st.sidebar.file_uploader("Upload locations.xlsx or .csv", type=["xlsx", "xls", "csv"])

st.sidebar.header("2. Upload Observed Duration Files")
obs_uploaded = {}
for dur in ["1h", "3h", "6h", "12h", "24h"]:
    obs_uploaded[dur] = st.sidebar.file_uploader(f"Observed {dur} file", type=["xlsx", "xls", "csv"], key=f"obs_{dur}")

st.sidebar.header("3. Upload ERA5-Land Duration Files")
era_uploaded = {}
for dur in ["1h", "3h", "6h", "12h", "24h"]:
    era_uploaded[dur] = st.sidebar.file_uploader(f"ERA5-Land {dur} file", type=["xlsx", "xls", "csv"], key=f"era_{dur}")

# Helper function to read Excel or CSV files from memory
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def load_and_deduplicate(uploaded_file):
    df = read_uploaded_file(uploaded_file)
    if df is None:
        return None
    # Assume first column or 'anno' is the index/year
    if 'anno' in [str(c).lower() for c in df.columns]:
        anno_col = [c for c in df.columns if str(c).lower() == 'anno'][0]
        df = df.set_index(anno_col)
    else:
        df = df.set_index(df.columns[0])
    
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

# Check if all essential files are uploaded
all_obs_ready = all(v is not None for v in obs_uploaded.values())
all_era_ready = all(v is not None for v in era_uploaded.values())

if not (locations_file and all_obs_ready and all_era_ready):
    st.info("👈 Please upload your station metadata file and all 5 duration AMP data for both Observed and ERA5-Land in the sidebar to begin.")
    st.stop()

# =========================================================
# 2. DATA PROCESSING, MAPPING & COMMON STATIONS
# =========================================================
df_meta = read_uploaded_file(locations_file)
code_col = [c for c in df_meta.columns if c.lower() in ['code', 'station_code', 'id', 'station']][0]
elev_col = [c for c in df_meta.columns if c.lower() in ['elev', 'elevation', 'alt', 'altitude', 'z']][0]
df_meta[code_col] = df_meta[code_col].astype(str).str.strip()

# Check if a region/zone column exists in metadata, otherwise map elevation bands
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

# Build mapping dictionaries
raw_elev_map = dict(zip(df_meta[code_col], df_meta[elev_col]))
if has_region_col:
    region_map = dict(zip(df_meta[code_col], df_meta[reg_col].astype(str)))
else:
    region_map = {stn: map_elevation_class(elev) for stn, elev in raw_elev_map.items()}

durations_str = ["1h", "3h", "6h", "12h", "24h"]
dur_hours = np.array([1, 3, 6, 12, 24])
thresholds = {"1h": 300, "3h": 500, "6h": 600, "12h": 800, "24h": 1000}
return_periods = [20, 50, 100, 200]

obs_dict = {dur: load_and_deduplicate(obs_uploaded[dur]) for dur in durations_str}
era_dict = {dur: load_and_deduplicate(era_uploaded[dur]) for dur in durations_str}

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

# Find common stations across all duration files and metadata
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

# Cache GEV calculation results across stations
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
    
    # Get unique regions/regimes available in the dataset
    available_regions = ['All'] + sorted(list(set(station_cache[s]['region'] for s in common_stations)))
    selected_region = st.selectbox("1. Select Region / Regime:", available_regions)
    
    # Filter stations based on selected region
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
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    for idx_T, T in enumerate(return_periods):
        obs_y = [stn_data["obs"][dur][idx_T] for dur in durations_str]
        era_y = [stn_data["era"][dur][idx_T] for dur in durations_str]
        
        # Plot Observed (Solid line with filled circle)
        ax.plot(dur_hours, obs_y, marker="o", linestyle="-", color=colors[idx_T],
                linewidth=1.5, markersize=5, label=f"Obs ({T} yr)")
        
        # Plot ERA5-Land (Dashed line with hollow square)
        ax.plot(dur_hours, era_y, marker="s", linestyle="--", color=colors[idx_T],
                linewidth=1.5, markersize=5, markerfacecolor="white", label=f"ERA5 ({T} yr)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dur_hours)
    ax.set_xticklabels([f"{h}h" for h in dur_hours], fontsize=9)
    ax.set_xlim(0.6, 30)
    
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.3)
    
    ax.set_title(f"Station: {selected_stn} | Region: {stn_data['region']} | Elev: {stn_data['elev']} m", fontsize=10, fontweight='bold')
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
            "ERA5_Land_mm": round(e_val, 2) if not np.isnan(e_val) else np.nan,
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
        mime="text/css",
    )

with tab2:
    col_obs, col_era = st.columns(2)
    
    with col_obs:
        st.markdown("**Observed Raw Return Levels (mm)**")
        df_obs_matrix = df_results.pivot(index="Duration", columns="Return_Period_yr", values="Observed_mm")
        st.dataframe(df_obs_matrix, use_container_width=True)
        
    with col_era:
        st.markdown("**ERA5-Land Raw Return Levels (mm)**")
        df_era_matrix = df_results.pivot(index="Duration", columns="Return_Period_yr", values="ERA5_Land_mm")
        st.dataframe(df_era_matrix, use_container_width=True)