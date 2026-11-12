import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import genextreme
import geopandas as gpd
import seaborn as sns
import io
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Can ERA5-Land data be used for Engineering purpose (Design Return Level)?", layout="wide")

st.title("🌧️ Can ERA5-Land data be used for Engineering purpose (Design Return Level)?")

# Methodology Note
st.info(
    "**Methodology:**\n\n"
    "To assess the suitability of ERA5-Land reanalysis data for urban infrastructure design and other engineering "
    "applications, we derived Duration-Depth-Frequency (DDF) curves for each station for both observed "
    "and ERA5-Land AMPs (1951-2023). The AMPs (observed and ERA5-Land) were fitted using the Generalized Extreme Value (GEV) "
    "distribution, and the DDF curves were computed for 1-, 3-, 6-, 12-, and 24-hr durations and return periods ($T$) "
    "of 20, 50, 100, and 200 years (Soto-Escobar et al., 2026). We then quantified relative bias, PBIAS, and KGE across storm duration "
    "and elevation classes."
)

# =========================================================
# 1. AUTOMATIC FILE LOADING FROM GITHUB REPOSITORY / PATHS
# =========================================================
durations_str = ["1h", "3h", "6h", "12h", "24h"]
dur_hours = np.array([1, 3, 6, 12, 24])
thresholds = {"1h": 300, "3h": 500, "6h": 600, "12h": 800, "24h": 1000}
return_periods = [20, 50, 100, 200]
elevation_labels = ["Low (< 230 m)", "Mid (230–720 m)", "High (> 720 m)"]

try:
    df_meta = pd.read_excel("locations.xlsx")
    
    obs_dict = {}
    era_dict = {}
    
    for dur in durations_str:
        obs_file = f"OBS_ARF_corrected_{dur}.xlsx"
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

# Load Shapefile for Study Area Map
try:
    shp_file = "italy_regions.shp"
    italy = gpd.read_file(shp_file).to_crs("EPSG:4326")
    regions_list = [
        "Piedmont", "Valle d'Aosta", "Liguria", "Lombardy", 
        "Veneto", "Friuli Venezia Giulia", "Emilia-Romagna", "Trentino-Alto Adige"
    ]
    region_col = 'name' if 'name' in italy.columns else 'DEN_REG'
    north_shp = italy[italy[region_col].isin(regions_list)]
except Exception:
    north_shp = None

# =========================================================
# 2. DATA PROCESSING & ELEVATION MAPPING
# =========================================================
code_col = [c for c in df_meta.columns if c.lower() in ['code', 'station_code', 'id', 'station']][0]
elev_col = [c for c in df_meta.columns if c.lower() in ['elev', 'elevation', 'alt', 'altitude', 'z']][0]
df_meta[code_col] = df_meta[code_col].astype(str).str.strip()

lat_candidates = [c for c in df_meta.columns if c.lower() in ['lat', 'latitude', 'y']]
lat_col = lat_candidates[0] if lat_candidates else None

lon_candidates = [c for c in df_meta.columns if c.lower() in ['lon', 'long', 'longitude', 'x']]
lon_col = lon_candidates[0] if lon_candidates else None

region_col_candidates = [c for c in df_meta.columns if c.lower() in ['region', 'zone', 'area', 'province', 'basin']]
has_region_col = len(region_col_candidates) > 0
reg_col = region_col_candidates[0] if has_region_col else None

df_meta['elevation_numeric'] = pd.to_numeric(df_meta[elev_col], errors='coerce')
bins = [-np.inf, 230, 720, np.inf]
df_meta['elevation_mapped'] = pd.cut(
    df_meta['elevation_numeric'], 
    bins=bins, 
    labels=elevation_labels, 
    include_lowest=True
)

raw_elev_map = dict(zip(df_meta[code_col], df_meta[elev_col]))
elevation_map = dict(zip(df_meta[code_col], df_meta['elevation_mapped']))
raw_lat_map = dict(zip(df_meta[code_col], df_meta[lat_col])) if lat_col else {}
raw_lon_map = dict(zip(df_meta[code_col], df_meta[lon_col])) if lon_col else {}

if has_region_col:
    region_map = dict(zip(df_meta[code_col], df_meta[reg_col].astype(str)))
else:
    region_map = {stn: str(elevation_map.get(stn, 'Unknown')) for stn in raw_elev_map.keys()}

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
            "lat": raw_lat_map.get(stn, "N/A"),
            "lon": raw_lon_map.get(stn, "N/A"),
            "region": region_map.get(stn, "Unknown"),
            "elevation_class": str(elevation_map.get(stn, "Unknown"))
        }
    return cache

station_cache = precompute_station_curves(common_stations)

# Vectorized metrics computation for all stations across durations
def compute_vectorized_metrics(df_obs, df_sim, elevation_map, duration):
    common_cols = df_obs.columns.intersection(df_sim.columns).intersection(pd.Index(elevation_map.keys()))
    obs = df_obs[common_cols]
    sim = df_sim[common_cols]
    
    valid_mask = obs.notna() & sim.notna()
    obs = obs.where(valid_mask)
    sim = sim.where(valid_mask)
    
    obs_sum = obs.sum(axis=0)
    sim_sum = sim.sum(axis=0)
    pbias = np.where(obs_sum != 0, 100.0 * (sim_sum - obs_sum) / obs_sum, np.nan)
    
    obs_mean = obs.mean(axis=0)
    sim_mean = sim.mean(axis=0)
    obs_std = obs.std(axis=0, ddof=1)
    sim_std = sim.std(axis=0, ddof=1)
    
    obs_centered = obs - obs_mean
    sim_centered = sim - sim_mean
    cov = (obs_centered * sim_centered).sum(axis=0)
    r = cov / ((obs.count(axis=0) - 1) * obs_std * sim_std)
    
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    
    records = []
    for i, col in enumerate(common_cols):
        records.append({
            'Duration': duration,
            'Station_Code': col,
            'Elevation_Class': elevation_map[col],
            'PBIAS': pbias[i],
            'KGE': kge[i]
        })
    return records

@st.cache_data
def generate_dataset_metrics():
    metrics_results = []
    for duration in durations_str:
        df_o = obs_dict[duration].align(era_dict[duration], join='inner', axis=0)[0]
        df_e = era_dict[duration].align(obs_dict[duration], join='inner', axis=0)[1]
        metrics_results.extend(compute_vectorized_metrics(df_o, df_e, elevation_map, duration))
    
    df_m = pd.DataFrame(metrics_results)
    df_m['Duration'] = pd.Categorical(df_m['Duration'], categories=durations_str, ordered=True)
    df_m['Elevation_Class'] = pd.Categorical(df_m['Elevation_Class'], categories=elevation_labels, ordered=True)
    return df_m.sort_values(by=['Duration', 'Elevation_Class']).reset_index(drop=True)

df_metrics = generate_dataset_metrics()

# =========================================================
# 3. INTERACTIVE DASHBOARD VIEW (REGION -> STATION SELECTOR)
# =========================================================
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Filter & Select")
    
    available_regions = ['All'] + sorted(list(set(station_cache[s]['region'] for s in common_stations)))
    selected_region = st.selectbox("1. Select Region:", available_regions)
    
    if selected_region == 'All':
        filtered_stations = common_stations
    else:
        filtered_stations = [s for s in common_stations if station_cache[s]['region'] == selected_region]
    
    if not filtered_stations:
        st.warning("No stations found in this region.")
        st.stop()
        
    selected_stn = st.selectbox("2. Select Station:", filtered_stations)
    
    st.info(
        f"**Selected Station:** `{selected_stn}`\n\n"
        f"**Region:** {station_cache[selected_stn]['region']}\n\n"
        f"**Elevation:** {station_cache[selected_stn]['elev']} m\n\n"
        f"**Elevation Class:** {station_cache[selected_stn]['elevation_class']}"
    )

    # Study Area Map Widget in Sidebar
    st.markdown("---")
    st.subheader("🗺️ Study Area Map")
    if north_shp is not None and lat_col and lon_col:
        fig_map, ax_map = plt.subplots(figsize=(4, 4), dpi=600)
        north_shp.plot(ax=ax_map, color="#f0f2f6", edgecolor="#cbd5e1", linewidth=0.8)
        
        ax_map.scatter(df_meta[lon_col], df_meta[lat_col], color="#94a3b8", s=10, alpha=0.6, label="Stations")
        
        sel_lat = float(station_cache[selected_stn]['lat'])
        sel_lon = float(station_cache[selected_stn]['lon'])
        ax_map.scatter([sel_lon], [sel_lat], color="#d95f02", s=35, marker="*", zorder=5, label=f"Selected ({selected_stn})")
        
        ax_map.set_title(f"Location Map: {selected_stn}", fontsize=8, fontweight='bold')
        ax_map.axis('off')
        ax_map.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
        st.pyplot(fig_map)
    else:
        st.info("Shapefile (`italy_regions.shp`) or coordinates not found for map rendering.")

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
        f"Station: {selected_stn} | Class: {stn_data['elevation_class']} | Elev: {stn_data['elev']} m | Lat: {stn_data['lat']}° | Lon: {stn_data['lon']}°",
        fontsize=10,
        fontweight='bold'
    )
    ax.set_xlabel("Duration", fontsize=9, fontweight='bold')
    ax.set_ylabel("Return Level (mm)", fontsize=9, fontweight='bold')
    ax.legend(frameon=True, fontsize=7, loc="upper left", ncol=2)
    
    st.pyplot(fig)

# =========================================================
# 4. DATASET-WIDE PUBLICATION PLOT & ELEVATION METRICS
# =========================================================
st.markdown("---")
st.subheader("📈 Dataset-Wide Elevation & Duration Performance (PBIAS & KGE)")

st.write(
    "Evaluation of overall model performance categorized across **Duration** and **Elevation Class** "
    "using Percent Bias ($\text{PBIAS}$) and Kling–Gupta Efficiency ($\text{KGE}$). "
    "Solid/dashed reference lines mark optimal performance thresholds ($\text{PBIAS} = 0\%$, $\text{KGE} = 1.0$). "
)

col_plot, col_summary_tbl = st.columns([3, 2])

with col_plot:
    elevation_palette = {
        elevation_labels[0]: "#2b83ba",
        elevation_labels[1]: "#fdae61",
        elevation_labels[2]: "#d7191c"
    }
    dark_grey = "#2b2b2b"

    fig_pub, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=300)
    plot_metrics = [
        {'key': 'PBIAS', 'label': 'Percent Bias (%)', 'ref': 0.0, 'letter': 'a'},
        {'key': 'KGE',   'label': 'Kling–Gupta Efficiency (KGE)', 'ref': 1.0, 'letter': 'b'}
    ]

    for idx, metric_info in enumerate(plot_metrics):
        ax = axes[idx]
        m_key = metric_info['key']
        m_label = metric_info['label']
        ref_line = metric_info['ref']
        
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, color='gray')
        
        sns.boxplot(
            data=df_metrics,
            x='Duration',
            y=m_key,
            hue='Elevation_Class',
            hue_order=elevation_labels,
            order=durations_str,
            width=0.6,
            showfliers=False,
            palette=elevation_palette,
            boxprops=dict(alpha=0.85, edgecolor=dark_grey, linewidth=0.8),
            whiskerprops=dict(color=dark_grey, linewidth=0.8),
            capprops=dict(color=dark_grey, linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.2),
            ax=ax
        )
        
        if ref_line is not None:
            ax.axhline(ref_line, color='black', linestyle='--', linewidth=0.9, zorder=1)
        
        ax.set_title(f"({metric_info['letter']})", loc='left', fontweight='bold', pad=6)
        ax.set_xlabel("Duration", fontweight='bold', labelpad=4)
        ax.set_ylabel(m_label, fontweight='bold', labelpad=4)
        
        if idx == 0:
            ax.legend(
                title="Elevation Class", 
                title_fontsize='7',
                frameon=True, 
                facecolor='white', 
                edgecolor='none',
                framealpha=0.85, 
                loc='upper left'
            )
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
                
        sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    st.pyplot(fig_pub)

with col_summary_tbl:
    st.markdown("**Median Metrics Table**")
    df_medians = df_metrics.groupby(['Duration', 'Elevation_Class'], observed=False)[['PBIAS', 'KGE']].median()
    pbias_pivot = df_medians['PBIAS'].unstack(level='Elevation_Class').reindex(durations_str).round(2)
    st.dataframe(pbias_pivot, use_container_width=True)

    # Download button for median metrics
    output_excel_buffer = io.BytesIO()
    with pd.ExcelWriter(output_excel_buffer, engine='openpyxl') as writer:
        pbias_pivot.to_excel(writer, sheet_name='PBIAS_Medians')
        df_metrics.to_excel(writer, sheet_name='Raw_Station_Metrics', index=False)
    
    st.download_button(
        label="📥 Download Elevation Summary (Excel)",
        data=output_excel_buffer.getvalue(),
        file_name="Median_Metrics_Duration_Elevation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

<FollowUp label="Want me to add an interactive correlation scatter plot between station elevation and PBIAS?" query="Yes, add an interactive scatter plot showing how PBIAS varies with station elevation across different durations."/>
