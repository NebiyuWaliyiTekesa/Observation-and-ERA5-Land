import os
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import io

# =========================================================
# PAGE CONFIGURATION & PUBLICATION STYLE SETTINGS
# =========================================================
st.set_page_config(
    page_title="ERA5-Land vs. Observed AMP Elevation KDE Analysis",
    layout="wide"
)

st.title("📊 PDF Analysis")

st.info(
    "**Methodology:**\n\n"
    "To assess the suitability of ERA5-Land reanalysis data for urban infrastructure design and engineering applications, "
    "this dashboard evaluates whether ERA5-Land captures the upper heavy tails of the Annual Maximum Precipitation (AMP) "  
    "using PDF analysis and other statistical summaries for individual stations or regions for each storm duration (1-, 3-, 6-, 12-, and 24-hr).")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

durations = ["1h", "3h", "6h", "12h", "24h"]

custom_x_limits = {
    "1h": 60,
    "3h": 150,
    "6h": 200,
    "12h": 300,
    "24h": 400
}

hue_order = ['Low (< 230 m)', 'Mid (230–720 m)', 'High (> 720 m)']

palette = {
    'Low (< 230 m)': "#2b83ba",       # Blue
    'Mid (230–720 m)': "#e6ab02",     # Gold / Amber
    'High (> 720 m)': "#d95f02"       # Vermillion / Red
}

# =========================================================
# 1. LOAD METADATA, GIS SHAPEFILES, & ARF DATA
# =========================================================
@st.cache_resource
def load_gis_and_metadata():
    try:
        df_meta = pd.read_excel("locations.xlsx")
        
        code_col = [c for c in df_meta.columns if c.lower() in ['code', 'station_code', 'id', 'station']][0]
        elev_col = [c for c in df_meta.columns if c.lower() in ['elev', 'elevation', 'alt', 'altitude', 'z']][0]
        lat_col = [c for c in df_meta.columns if c.lower() in ['lat', 'latitude', 'y']][0]
        lon_col = [c for c in df_meta.columns if c.lower() in ['lon', 'longitude', 'long', 'x']][0]
        region_col = [c for c in df_meta.columns if c.lower() in ['region', 'province', 'area', 'zone']][0] if any(c.lower() in ['region', 'province', 'area', 'zone'] for c in df_meta.columns) else None

        df_meta[code_col] = df_meta[code_col].astype(str).str.strip()

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
                return np.nan

        df_meta['elev_mapped'] = df_meta[elev_col].apply(map_elevation_class)

        shapefile_path = None
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".shp"):
                    shapefile_path = os.path.join(root, file)
                    break
            if shapefile_path:
                break
        
        gdf_boundary = gpd.read_file(shapefile_path) if shapefile_path else None
        
        return df_meta, code_col, elev_col, lat_col, lon_col, region_col, gdf_boundary
    except Exception as e:
        st.error(f"Error loading metadata or shapefiles: {e}")
        return None, None, None, None, None, None, None

df_meta, code_col, elev_col, lat_col, lon_col, region_col, gdf_boundary = load_gis_and_metadata()

if df_meta is None:
    st.stop()

elev_map = dict(zip(df_meta[code_col], df_meta['elev_mapped']))

@st.cache_data
def load_precipitation_data():
    obs_dicts = {}
    era_dicts = {}
    for d in durations:
        obs_file = f"OBS_ARF_corrected_{d}.xlsx"
        num_str = d.replace('h', '')
        era_file = f"{num_str}hr.xlsx"
        
        obs_dicts[d] = pd.read_excel(obs_file, index_col=0)
        era_dicts[d] = pd.read_excel(era_file, index_col=0)
        
        obs_dicts[d].columns = obs_dicts[d].columns.astype(str).str.strip()
        era_dicts[d].columns = era_dicts[d].columns.astype(str).str.strip()
    return obs_dicts, era_dicts

obs_dicts, era_dicts = load_precipitation_data()

# =========================================================
# 2. SIDEBAR CONTROLS & STUDY AREA MAP
# =========================================================
st.sidebar.header("Filter & Select")

all_regions = sorted(df_meta[region_col].dropna().unique().tolist()) if region_col else []
selected_region = st.sidebar.selectbox("1. Select Region:", ["All"] + all_regions)

if selected_region != "All" and region_col:
    filtered_stations_df = df_meta[df_meta[region_col] == selected_region]
else:
    filtered_stations_df = df_meta

available_stations = sorted(filtered_stations_df[code_col].unique().tolist())
selected_station = st.sidebar.selectbox("2. Select Station:", ["All Stations"] + available_stations)

if selected_station != "All Stations":
    st_row = df_meta[df_meta[code_col] == selected_station].iloc[0]
    st.sidebar.markdown(f"""
    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; color: #31333F;">
        <b>Selected Station:</b> {selected_station}<br>
        <b>Region:</b> {st_row[region_col] if region_col else 'N/A'}<br>
        <b>Elevation:</b> {st_row[elev_col]} m ({st_row['elev_mapped']})
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Study Area Map")

fig_map, ax_map = plt.subplots(figsize=(4, 4), dpi=300)
if gdf_boundary is not None:
    gdf_boundary.plot(ax=ax_map, color='#f7f7f7', edgecolor='#bdbdbd', linewidth=0.8)

ax_map.scatter(df_meta[lon_col], df_meta[lat_col], color='#9ecae1', s=12, edgecolor='none', alpha=0.7, label='Stations')

if selected_station != "All Stations":
    sel_row = df_meta[df_meta[code_col] == selected_station].iloc[0]
    ax_map.scatter([sel_row[lon_col]], [sel_row[lat_col]], color='#d95f02', s=45, marker='*', edgecolor='black', linewidth=0.5, label='Selected Station')
    ax_map.set_title(f"Station: {selected_station}", fontsize=9, fontweight='bold')
elif selected_region != "All" and region_col:
    reg_df = df_meta[df_meta[region_col] == selected_region]
    ax_map.scatter(reg_df[lon_col], reg_df[lat_col], color='#2b83ba', s=20, edgecolor='black', linewidth=0.3, label=selected_region)
    ax_map.set_title(f"Region: {selected_region}", fontsize=9, fontweight='bold')
else:
    ax_map.set_title("Northern Italy Stations", fontsize=9, fontweight='bold')

ax_map.axis('off')
st.sidebar.pyplot(fig_map)

# =========================================================
# 3. DATA PROCESSING & SUMMARY STATISTICS COMPUTATION
# =========================================================
@st.cache_data
def process_filtered_data(sel_region, sel_station):
    pooled_data = {}
    summary_rows = []

    for duration in durations:
        df_obs = obs_dicts[duration]
        df_era = era_dicts[duration]
        
        if sel_station != "All Stations":
            target_stations = [sel_station]
        elif sel_region != "All" and region_col:
            target_stations = df_meta[df_meta[region_col] == sel_region][code_col].tolist()
        else:
            target_stations = df_meta[code_col].tolist()

        common_stations = [c for c in df_obs.columns if c in df_era.columns and c in elev_map and c in target_stations]
        thr = custom_x_limits[duration]
        
        if sel_station != "All Stations":
            # For a single station, group by its specific elevation category
            st_cat = elev_map.get(sel_station, 'Low (< 230 m)')
            obs_groups = {grp: [] for grp in hue_order}
            era_groups = {grp: [] for grp in hue_order}
            
            if common_stations:
                st_code = common_stations[0]
                o_s = df_obs[st_code].dropna()
                e_s = df_era[st_code].dropna()
                idx = o_s.index.intersection(e_s.index)
                o_s, e_s = o_s.loc[idx], e_s.loc[idx]
                valid_mask = (o_s <= thr) & (e_s <= thr)
                o_s, e_s = o_s[valid_mask], e_s[valid_mask]
                
                if len(o_s) > 0 and pd.notna(st_cat):
                    obs_groups[st_cat].extend(o_s.values)
                    era_groups[st_cat].extend(e_s.values)
        else:
            obs_groups = {grp: [] for grp in hue_order}
            era_groups = {grp: [] for grp in hue_order}
            
            for st_code in common_stations:
                cat = elev_map[st_code]
                if pd.isna(cat):
                    continue
                
                o_s = df_obs[st_code].dropna()
                e_s = df_era[st_code].dropna()
                
                idx = o_s.index.intersection(e_s.index)
                o_s, e_s = o_s.loc[idx], e_s.loc[idx]
                
                valid_mask = (o_s <= thr) & (e_s <= thr)
                o_s, e_s = o_s[valid_mask], e_s[valid_mask]
                
                if len(o_s) > 0:
                    obs_groups[cat].extend(o_s.values)
                    era_groups[cat].extend(e_s.values)

        pooled_data[duration] = {
            'OBS': {grp: np.array(vals) for grp, vals in obs_groups.items()},
            'ERA5-Land': {grp: np.array(vals) for grp, vals in era_groups.items()}
        }

        active_groups = [elev_map[sel_station]] if sel_station != "All Stations" and sel_station in elev_map else hue_order
        for grp in active_groups:
            for source, data_dict in [('OBS', obs_groups), ('ERA5-Land', era_groups)]:
                arr = np.array(data_dict[grp])
                if len(arr) > 0:
                    summary_rows.append({
                        'Duration': duration,
                        'Elevation Class': grp,
                        'Dataset Source': source,
                        'Sample Count (N)': len(arr),
                        'Mean (mm)': round(np.mean(arr), 2),
                        'Std Dev (mm)': round(np.std(arr, ddof=1), 2),
                        'Skewness': round(float(skew(arr)), 3),
                        'Kurtosis': round(float(kurtosis(arr)), 3),
                        'P90 (mm)': round(np.percentile(arr, 90), 2),
                        'P99 (mm)': round(np.percentile(arr, 99), 2)
                    })

    return pooled_data, pd.DataFrame(summary_rows)

pooled_data, df_summary = process_filtered_data(selected_region, selected_station)

# =========================================================
# 4. DISPLAY SUMMARY TABLE & DOWNLOAD OPTION
# =========================================================
filter_title_suffix = f"(Region: {selected_region}, Station: {selected_station})" if selected_station != "All Stations" else (f"(Region: {selected_region})" if selected_region != "All" else "(All Regions & Stations)")
st.subheader(f"📋 Summary Statistics {filter_title_suffix}")

if df_summary.empty:
    st.warning("No data available for the selected region/station filter configuration.")
else:
    st.dataframe(df_summary, use_container_width=True)

    csv_buffer = io.BytesIO()
    df_summary.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Filtered Summary Statistics (CSV)",
        data=csv_buffer.getvalue(),
        file_name="PDF_Summary_Statistics_Filtered.csv",
        mime="text/csv"
    )

st.markdown("---")

# =========================================================
# 5. PLOTTING 2x3 KDE GRID
# =========================================================
plot_title_suffix = f"for Station {selected_station}" if selected_station != "All Stations" else (f"for Region {selected_region}" if selected_region != "All" else "Across All Elevations & Durations")
st.subheader(f"📈 Probability Density Functions (KDE) {plot_title_suffix}")

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=600)
plt.subplots_adjust(wspace=0.15, hspace=0.15, right=0.96, left=0.06, top=0.95, bottom=0.08)

active_plot_groups = [elev_map[selected_station]] if selected_station != "All Stations" and selected_station in elev_map else hue_order

for i in range(6):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    
    if i < 5:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')

        dur = durations[i]
        
        for grp in active_plot_groups:
            obs_arr = pooled_data[dur]['OBS'][grp]
            era_arr = pooled_data[dur]['ERA5-Land'][grp]
            
            if len(obs_arr) > 1:
                sns.kdeplot(
                    obs_arr, ax=ax, color=palette[grp], linestyle='-', 
                    linewidth=1.6, label=f'{grp} OBS', cut=0
                )
            
            if len(era_arr) > 1:
                sns.kdeplot(
                    era_arr, ax=ax, color=palette[grp], linestyle='--', 
                    linewidth=1.6, label=f'{grp} ERA5-Land', cut=0
                )
        
        x_limit = custom_x_limits[dur]
        ax.set_xlim(left=0, right=x_limit)
        ax.set_ylim(bottom=0)
        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        
        ax.text(0.2, 0.88, f"({chr(97+i)}) {dur}", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.7))

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("AMP (mm)", fontweight='bold')
            
        if col == 0:
            ax.set_ylabel("Probability Density", fontweight='bold')
        else:
            ax.set_ylabel("")

    else:
        ax.set_axis_off()
        if selected_station != "All Stations" and selected_station in elev_map:
            st_cat = elev_map[selected_station]
            legend_elements = [
                Line2D([0], [0], color='black', lw=2, linestyle='-', label=f'OBS ({selected_station})'),
                Line2D([0], [0], color='black', lw=2, linestyle='--', label=f'ERA5-Land ({selected_station})'),
                Line2D([0], [0], color=palette[st_cat], lw=2, label=f'Class: {st_cat}')
            ]
        else:
            legend_elements = [
                Line2D([0], [0], color='black', lw=2, linestyle='-', label='OBS'),
                Line2D([0], [0], color='black', lw=2, linestyle='--', label='ERA5-Land'),
                Line2D([0], [0], color=palette['Low (< 230 m)'], lw=2, label='Low Elevation (< 230 m)'),
                Line2D([0], [0], color=palette['Mid (230–720 m)'], lw=2, label='Mid Elevation (230–720 m)'),
                Line2D([0], [0], color=palette['High (> 720 m)'], lw=2, label='High Elevation (> 720 m)')
            ]
        ax.legend(handles=legend_elements, loc='center', fontsize=9.5, frameon=False, edgecolor='black')

st.pyplot(fig)

# =========================================================
# 6. ENGINEERING ANALYSIS & CONCLUSION
# =========================================================
st.markdown("---")
st.subheader("📌 Engineering Analysis & Conclusions")

st.markdown("""
* **Topographic Control on Convective Extremes:** Stratifying AMP series by elevation or analyzing individual stations reveals distinct distributional shifts, highlighting how complex terrain modulates sub-daily convective storm intensity.
* **Reanalysis Performance & Bias:** ERA5-Land exhibits varying degrees of fidelity relative to station gauges depending on elevation band and storm duration, particularly underestimating extreme percentiles ($P_{99}$) in complex high-elevation terrain due to grid-cell smoothing effects.
* **Implications for Infrastructure Design:** Because generalized regional summaries fail to capture local microclimatic extremes, rigorous station-based validation is non-negotiable for reliable urban drainage and flood mitigation planning (Soto-Escobar et al., 2026).
""")
