import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="ERA5-Land vs Observations: Station & Regional Evaluation", layout="wide")

st.title("🌧️ ERA5-Land vs Observations: Station & Regional Evaluation")

# Methodology Note
st.info(
    "**Methodology:**\n\n"
    "To assess the suitability of ERA5-Land reanalysis precipitation data against ground station observations "
    "across Northern Italy (1951-2023), we perform rigorous temporal alignment per station. "
    "Continuous evaluation metrics—**Percent Bias (PBIAS)** and **Kling–Gupta Efficiency (KGE)**—are "
    "calculated exactly for each station and aggregated across storm durations (1h, 3h, 6h, 12h, 24h), "
    "elevation classes, and administrative regions."
)

# =========================================================
# 1. AUTOMATIC FILE LOADING FROM REPOSITORY
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
        df_obs.index = pd.to_numeric(df_obs.index, errors='coerce')
        df_obs.columns = df_obs.columns.astype(str).str.strip()
        obs_dict[dur] = df_obs.loc[df_obs.index.notna(), ~df_obs.columns.duplicated(keep="first")]
        
        # Load and deduplicate ERA5 file
        df_era = pd.read_excel(era_file)
        anno_col_era = [c for c in df_era.columns if str(c).lower() in ['anno', 'year', 'date']][0] if any(str(c).lower() in ['anno', 'year', 'date'] for c in df_era.columns) else df_era.columns[0]
        df_era = df_era.set_index(anno_col_era)
        df_era.index = pd.to_numeric(df_era.index, errors='coerce')
        df_era.columns = df_era.columns.astype(str).str.strip()
        era_dict[dur] = df_era.loc[df_era.index.notna(), ~df_era.columns.duplicated(keep="first")]

except Exception as e:
    st.error(f"Error loading files automatically from repository: {e}")
    st.stop()

# =========================================================
# 2. METADATA PROCESSING & MAPPING
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
# 3. ROBUST EXACT METRICS CALCULATION (PER STATION)
# =========================================================
@st.cache_data
def compute_all_station_metrics():
    records = []
    for dur in durations_str:
        df_obs = obs_dict[dur]
        df_era = era_dict[dur]
        
        for stn in common_stations:
            if stn not in df_obs.columns or stn not in df_era.columns:
                continue
            
            sub_obs = df_obs[stn]
            sub_era = df_era[stn]
            aligned = pd.concat([sub_obs, sub_era], axis=1, join='inner').dropna()
            
            if len(aligned) < 3:
                continue
            
            o = aligned.iloc[:, 0]
            e = aligned.iloc[:, 1]
            
            obs_sum = o.sum()
            sim_sum = e.sum()
            pbias = 100.0 * (sim_sum - obs_sum) / obs_sum if obs_sum != 0 else np.nan
            
            obs_mean = o.mean()
            sim_mean = e.mean()
            obs_std = o.std(ddof=1)
            sim_std = e.std(ddof=1)
            
            if obs_std > 0 and sim_std > 0 and obs_mean != 0:
                r = o.corr(e)
                alpha = sim_std / obs_std
                beta = sim_mean / obs_mean
                kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
            else:
                kge = np.nan
            
            records.append({
                'Duration': dur,
                'Station_Code': stn,
                'Region': region_map.get(stn, 'Unknown'),
                'Elevation_Class': str(elevation_map.get(stn, 'Unknown')),
                'Elevation_m': raw_elev_map.get(stn, np.nan),
                'Valid_Years': len(aligned),
                'PBIAS': round(pbias, 2) if not np.isnan(pbias) else np.nan,
                'KGE': round(kge, 3) if not np.isnan(kge) else np.nan
            })
            
    df_m = pd.DataFrame(records)
    if not df_m.empty:
        df_m['Duration'] = pd.Categorical(df_m['Duration'], categories=durations_str, ordered=True)
        df_m['Elevation_Class'] = pd.Categorical(df_m['Elevation_Class'], categories=elevation_labels, ordered=True)
    return df_m

df_metrics = compute_all_station_metrics()

if df_metrics.empty:
    st.error("No overlapping valid data records were found between Observations and ERA5-Land.")
    st.stop()

# =========================================================
# 4. INTERACTIVE STATION INSPECTOR & DUAL SUBPLOT
# =========================================================
st.markdown("---")
st.subheader("🔍 Station-Specific Performance Inspection")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    selected_reg = st.selectbox("Select Region:", ['All'] + sorted(list(df_metrics['Region'].dropna().unique())))
    
filtered_stns = common_stations if selected_reg == 'All' else sorted(list(df_metrics[df_metrics['Region'] == selected_reg]['Station_Code'].unique()))

with col_sel2:
    selected_station = st.selectbox("Select Station Code:", filtered_stns)

stn_df = df_metrics[df_metrics['Station_Code'] == selected_station].sort_values('Duration')

st.markdown(f"**Metrics Summary for Station `{selected_station}`:**")
st.dataframe(stn_df[['Duration', 'Elevation_Class', 'Elevation_m', 'Valid_Years', 'PBIAS', 'KGE']], use_container_width=True)

# Station Plot: Two separate subplots stacked vertically (1 for PBIAS, 1 for KGE)
fig_stn, (ax_pb, ax_kg) = plt.subplots(2, 1, figsize=(8, 5), dpi=300, sharex=True)

# PBIAS Subplot
ax_pb.plot(stn_df['Duration'], stn_df['PBIAS'], marker='o', linestyle='-', color='#d95f02', linewidth=1.8, label='PBIAS (%)')
ax_pb.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax_pb.set_ylabel("PBIAS (%)", fontweight='bold', color='#d95f02')
ax_pb.set_title(f"Station {selected_station} — Evaluation Metrics Across Durations", fontweight='bold', fontsize=10)
ax_pb.grid(True, linestyle=':', alpha=0.6)
ax_pb.legend(loc='upper right', frameon=True, fontsize=8)

# KGE Subplot
ax_kg.plot(stn_df['Duration'], stn_df['KGE'], marker='s', linestyle='--', color='#2b83ba', linewidth=1.8, label='KGE')
ax_kg.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
ax_kg.set_ylabel("Kling-Gupta Efficiency", fontweight='bold', color='#2b83ba')
ax_kg.set_xlabel("Duration", fontweight='bold')
ax_kg.set_ylim(-1.1, 1.1)
ax_kg.grid(True, linestyle=':', alpha=0.6)
ax_kg.legend(loc='upper right', frameon=True, fontsize=8)

plt.tight_layout()
st.pyplot(fig_stn)

# =========================================================
# 5. REGIONAL & ELEVATION MEDIAN AGGREGATES
# =========================================================
st.markdown("---")
st.subheader("📊 Regional & Elevation Performance Aggregates (Medians)")

tab_reg, tab_elev = st.tabs(["🌍 By Region", "⛰️ By Elevation Class"])

with tab_reg:
    df_region_medians = df_metrics.groupby(['Duration', 'Region'], observed=False)[['PBIAS', 'KGE']].median().unstack(level='Region').round(2)
    st.dataframe(df_region_medians, use_container_width=True)

with tab_elev:
    df_elev_medians = df_metrics.groupby(['Duration', 'Elevation_Class'], observed=False)[['PBIAS', 'KGE']].median().unstack(level='Elevation_Class').round(2)
    st.dataframe(df_elev_medians, use_container_width=True)

# =========================================================
# 6. PUBLICATION-READY BOXPLOTS (ELEVATION & REGIONAL)
# =========================================================
st.markdown("---")
st.subheader("📈 Dataset-Wide Publication Boxplots")

elevation_palette = {
    elevation_labels[0]: "#2b83ba",
    elevation_labels[1]: "#fdae61",
    elevation_labels[2]: "#d7191c"
}
dark_grey = "#2b2b2b"

# --- A. Elevation-Based Boxplots ---
st.markdown("**1. Performance by Elevation Class**")
fig_pub_elev, axes_elev = plt.subplots(1, 2, figsize=(8.0, 3.5), dpi=300)
plot_metrics = [
    {'key': 'PBIAS', 'label': 'Percent Bias (%)', 'ref': 0.0, 'letter': 'a'},
    {'key': 'KGE',   'label': 'Kling–Gupta Efficiency (KGE)', 'ref': 1.0, 'letter': 'b'}
]

for idx, metric_info in enumerate(plot_metrics):
    ax = axes_elev[idx]
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
        ax.legend(title="Elevation Class", title_fontsize='7', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85, loc='upper left')
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()
st.pyplot(fig_pub_elev)


# --- B. Region-Based Boxplots ---
st.markdown("**2. Performance by Region**")
fig_pub_reg, axes_reg = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=300)
available_regions = sorted(list(df_metrics['Region'].dropna().unique()))

for idx, metric_info in enumerate(plot_metrics):
    ax = axes_reg[idx]
    m_key = metric_info['key']
    m_label = metric_info['label']
    ref_line = metric_info['ref']
    
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, color='gray')
    
    sns.boxplot(
        data=df_metrics,
        x='Duration',
        y=m_key,
        hue='Region',
        order=durations_str,
        width=0.7,
        showfliers=False,
        palette="Set2",
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
        ax.legend(title="Region", title_fontsize='7', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
    sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()
st.pyplot(fig_pub_reg)


# --- C. Export Workbook ---
st.markdown("---")
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
