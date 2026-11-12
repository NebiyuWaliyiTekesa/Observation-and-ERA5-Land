import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="ERA5-Land vs Observations: PBIAS & KGE Evaluation", layout="wide")

st.title("🌧️ ERA5-Land vs Observations: PBIAS & KGE Evaluation")

# Methodology Note
st.info(
    "**Methodology:**\n\n"
    "To assess the suitability of ERA5-Land reanalysis precipitation data against ground station observations "
    "across Northern Italy (1951-2023), we compute continuous evaluation metrics—**Percent Bias (PBIAS)** "
    "and **Kling–Gupta Efficiency (KGE)**—disaggregated across storm durations (1h, 3h, 6h, 12h, 24h), "
    "elevation classes, and administrative regions."
)

# =========================================================
# 1. AUTOMATIC FILE LOADING FROM GITHUB REPOSITORY
# =========================================================
durations_str = ["1h", "3h", "6h", "12h", "24h"]
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
        anno_col_obs = [c for c in df_obs.columns if str(c).lower() in ['anno', 'year', 'date']][0] if any(str(c).lower() in ['anno', 'year', 'date'] for c in df_obs.columns) else df_obs.columns[0]
        df_obs = df_obs.set_index(anno_col_obs)
        df_obs.columns = df_obs.columns.astype(str).str.strip()
        obs_dict[dur] = df_obs.loc[:, ~df_obs.columns.duplicated(keep="first")]
        
        # Load and deduplicate ERA5 file
        df_era = pd.read_excel(era_file)
        anno_col_era = [c for c in df_era.columns if str(c).lower() in ['anno', 'year', 'date']][0] if any(str(c).lower() in ['anno', 'year', 'date'] for c in df_era.columns) else df_era.columns[0]
        df_era = df_era.set_index(anno_col_era)
        df_era.columns = df_era.columns.astype(str).str.strip()
        era_dict[dur] = df_era.loc[:, ~df_era.columns.duplicated(keep="first")]

except Exception as e:
    st.error(f"Error loading files automatically from repository: {e}")
    st.stop()

# =========================================================
# 2. METADATA PROCESSING & REGIONAL/ELEVATION MAPPING
# =========================================================
code_col = [c for c in df_meta.columns if c.lower() in ['code', 'station_code', 'id', 'station']][0]
elev_col = [c for c in df_meta.columns if c.lower() in ['elev', 'elevation', 'alt', 'altitude', 'z']][0]
df_meta[code_col] = df_meta[code_col].astype(str).str.strip()

region_col_candidates = [c for c in df_meta.columns if c.lower() in ['region', 'zone', 'area', 'province', 'basin', 'den_reg', 'name']]
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

if has_region_col:
    region_map = dict(zip(df_meta[code_col], df_meta[reg_col].astype(str)))
else:
    region_map = {stn: "Northern Italy" for stn in raw_elev_map.keys()}

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
    st.error("No matching station codes found across metadata and duration files. Check column headers.")
    st.stop()

# =========================================================
# 3. VECTORIZED EVALUATION METRICS (PBIAS & KGE)
# =========================================================
def compute_vectorized_metrics(df_obs, df_sim, elevation_map, region_map, duration):
    common_cols = df_obs.columns.intersection(df_sim.columns).intersection(pd.Index(elevation_map.keys()))
    obs = df_obs[common_cols]
    sim = df_sim[common_cols]
    
    valid_mask = obs.notna() & sim.notna()
    obs = obs.where(valid_mask)
    sim = sim.where(valid_mask)
    
    # PBIAS Calculation
    obs_sum = obs.sum(axis=0)
    sim_sum = sim.sum(axis=0)
    pbias = np.where(obs_sum != 0, 100.0 * (sim_sum - obs_sum) / obs_sum, np.nan)
    
    # KGE Calculation
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
            'Region': region_map.get(col, 'Unknown'),
            'Elevation_Class': elevation_map[col],
            'Elevation_m': raw_elev_map.get(col, np.nan),
            'PBIAS': pbias[i] if isinstance(pbias, np.ndarray) else pbias.loc[col],
            'KGE': kge.loc[col] if hasattr(kge, 'loc') else kge[i]
        })
    return records

@st.cache_data
def generate_dataset_metrics():
    metrics_results = []
    for duration in durations_str:
        df_o = obs_dict[duration].align(era_dict[duration], join='inner', axis=0)[0]
        df_e = era_dict[duration].align(obs_dict[duration], join='inner', axis=0)[1]
        metrics_results.extend(compute_vectorized_metrics(df_o, df_e, elevation_map, region_map, duration))
    
    df_m = pd.DataFrame(metrics_results)
    df_m['Duration'] = pd.Categorical(df_m['Duration'], categories=durations_str, ordered=True)
    df_m['Elevation_Class'] = pd.Categorical(df_m['Elevation_Class'], categories=elevation_labels, ordered=True)
    return df_m.sort_values(by=['Duration', 'Region', 'Elevation_Class']).reset_index(drop=True)

df_metrics = generate_dataset_metrics()

# =========================================================
# 4. DASHBOARD VIEW: FILTERING & STATION METRICS EXPLORER
# =========================================================
st.subheader("🔍 Station-Level Performance Explorer")

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_region_filter = st.selectbox(
        "Filter by Region:", 
        ['All'] + sorted(list(df_metrics['Region'].dropna().unique()))
    )
with col_f2:
    selected_dur_filter = st.selectbox(
        "Filter by Duration:", 
        ['All'] + durations_str
    )

df_filtered_stations = df_metrics.copy()
if selected_region_filter != 'All':
    df_filtered_stations = df_filtered_stations[df_filtered_stations['Region'] == selected_region_filter]
if selected_dur_filter != 'All':
    df_filtered_stations = df_filtered_stations[df_filtered_stations['Duration'] == selected_dur_filter]

st.dataframe(df_filtered_stations, use_container_width=True)

# =========================================================
# 5. REGIONAL & ELEVATION MEDIAN AGGREGATES
# =========================================================
st.markdown("---")
st.subheader("📊 Regional & Elevation Performance Aggregates (Medians)")

tab_reg, tab_elev = st.tabs(["🌍 By Region", "⛰️ By Elevation Class"])

with tab_reg:
    df_region_medians = df_metrics.groupby(['Duration', 'Region'], observed=False)[['PBIAS', 'KGE']].median().unstack(level='Region').round(2)
    st.write("Median PBIAS (%) and KGE grouped by Region across durations:")
    st.dataframe(df_region_medians, use_container_width=True)

with tab_elev:
    df_elev_medians = df_metrics.groupby(['Duration', 'Elevation_Class'], observed=False)[['PBIAS', 'KGE']].median().unstack(level='Elevation_Class').round(2)
    st.write("Median PBIAS (%) and KGE grouped by Elevation Class across durations:")
    st.dataframe(df_elev_medians, use_container_width=True)

# =========================================================
# 6. PUBLICATION-READY FIGURE & EXCEL DOWNLOAD
# =========================================================
st.markdown("---")
st.subheader("📈 Publication-Ready Performance Boxplots (Duration vs Elevation Class)")

col_plot, col_dl = st.columns([3, 1])

with col_plot:
    elevation_palette = {
        elevation_labels[0]: "#2b83ba",
        elevation_labels[1]: "#fdae61",
        elevation_labels[2]: "#d7191c"
    }
    dark_grey = "#2b2b2b"

    fig_pub, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), dpi=300)
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

with col_dl:
    st.markdown("**Export Results**")
    st.write("Download complete computed station-level metrics and regional summaries as an Excel workbook.")
    
    output_excel_buffer = io.BytesIO()
    with pd.ExcelWriter(output_excel_buffer, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Raw_Station_Metrics', index=False)
        df_region_medians.to_excel(writer, sheet_name='Regional_Medians')
        df_elev_medians.to_excel(writer, sheet_name='Elevation_Medians')
    
    st.download_button(
        label="📥 Download Complete Metrics (Excel)",
        data=output_excel_buffer.getvalue(),
        file_name="ERA5_Land_PBIAS_KGE_Evaluation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
