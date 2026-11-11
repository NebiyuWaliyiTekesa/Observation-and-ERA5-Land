import streamlit as st
import pandas as pd
import numpy as np
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

st.title("📊 ERA5-Land vs. Observed AMP Elevation-Stratified KDE Analysis")

st.info(
    "**Methodology & Scope:**\n\n"
    "This dashboard evaluates Annual Maximum Precipitation (AMP) distributions stratified across three elevation classes "
    "(`Low < 230 m`, `Mid 230–720 m`, `High > 720 m`) for storm durations ranging from 1 hour to 24 hours. "
    "Probability Density Functions (KDEs) and statistical summaries are computed using quality-controlled station observations "
    "and ERA5-Land reanalysis datasets."
)

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
# 1. AUTOMATED FILE LOADING & METADATA MAPPING
# =========================================================
try:
    df_meta = pd.read_excel("locations.xlsx")
    
    code_col = [c for c in df_meta.columns if c.lower() in ['code', 'station_code', 'id', 'station']][0]
    elev_col = [c for c in df_meta.columns if c.lower() in ['elev', 'elevation', 'alt', 'altitude', 'z']][0]
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
    elev_map = dict(zip(df_meta[code_col], df_meta['elev_mapped']))
    
    if region_col:
        region_map = dict(zip(df_meta[code_col], df_meta[region_col]))
    else:
        region_map = {code: "All Regions" for code in df_meta[code_col]}

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

except Exception as e:
    st.error(f"Error loading metadata or input files from repository: {e}")
    st.stop()

# =========================================================
# 2. SIDEBAR FILTER & SELECTION CONTROLS
# =========================================================
st.sidebar.header("Filter & Select")

# Region Selection
all_regions = sorted(df_meta[region_col].dropna().unique().tolist()) if region_col else []
selected_region = st.sidebar.selectbox("1. Select Region:", ["All"] + all_regions)

# Filter stations based on region choice
if selected_region != "All" and region_col:
    filtered_stations_df = df_meta[df_meta[region_col] == selected_region]
else:
    filtered_stations_df = df_meta

available_stations = sorted(filtered_stations_df[code_col].unique().tolist())
selected_station = st.sidebar.selectbox("2. Select Station:", ["All Stations"] + available_stations)

# Show Station Details Card if a specific station is picked
if selected_station != "All Stations":
    st_row = df_meta[df_meta[code_col] == selected_station].iloc[0]
    st.sidebar.markdown(f"""
    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; color: #31333F;">
        <b>Selected Station:</b> {selected_station}<br>
        <b>Region:</b> {st_row[region_col] if region_col else 'N/A'}<br>
        <b>Elevation:</b> {st_row[elev_col]} m ({st_row['elev_mapped']})
    </div>
    """, unsafe_allow_html=True)

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
        
        # Determine valid stations based on sidebar filters
        if sel_station != "All Stations":
            target_stations = [sel_station]
        elif sel_region != "All" and region_col:
            target_stations = df_meta[df_meta[region_col] == sel_region][code_col].tolist()
        else:
            target_stations = df_meta[code_col].tolist()

        common_stations = [c for c in df_obs.columns if c in df_era.columns and c in elev_map and c in target_stations]
        thr = custom_x_limits[duration]
        
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

        for grp in hue_order:
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
st.subheader(f"📋 Elevation-Stratified Summary Statistics {filter_title_suffix}")

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
st.subheader("📈 Probability Density Functions (KDE) Across Durations & Elevations")

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=600)
plt.subplots_adjust(wspace=0.15, hspace=0.15, right=0.96, left=0.06, top=0.95, bottom=0.08)

for i in range(6):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    
    if i < 5:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')

        dur = durations[i]
        
        for grp in hue_order:
            obs_arr = pooled_data[dur]['OBS'][grp]
            era_arr = pooled_data[dur]['ERA5-Land'][grp]
            
            if len(obs_arr) > 1: # KDE requires at least 2 points
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
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, linestyle='-', label='OBS'),
            Line2D([0], [0], color='black', lw=2, linestyle='--', label='ERA5-Land'),
            Line2D([0], [0], color=palette['Low (< 230 m)'], lw=2, label='Low Elevation (< 230 m)'),
            Line2D([0], [0], color=palette['Mid (230–720 m)'], lw=2, label='Mid Elevation (230–720 m)'),
            Line2D([0], [0], color=palette['High (> 720 m)'], lw=2, label='High Elevation (> 720 m)')
        ]
        ax.legend(handles=legend_elements, loc='center', fontsize=9.5, frameon=False, edgecolor='black')

st.pyplot(fig)
