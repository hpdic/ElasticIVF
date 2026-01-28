"""
plot_sivf_search.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for SIVF search performance benchmark.
Generates a 1x3 composite figure (Horizontal) with OVERSIZED fonts
to survive extreme scaling (fitting 3 plots into a single column width).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. Input Benchmark Data (2026-01-13: 100k/200k/500k)
# ==========================================
data = [
    # NB, nlist, System, QPS
    # --- 100k Vectors ---
    (100000, 1024, 'SIVF', 265258),
    (100000, 1024, 'Vanilla', 495769),
    (100000, 4096, 'SIVF', 382763),
    (100000, 4096, 'Vanilla', 560801),
    (100000, 16384, 'SIVF', 295321),
    (100000, 16384, 'Vanilla', 322907),

    # --- 200k Vectors ---
    (200000, 1024, 'SIVF', 157573),
    (200000, 1024, 'Vanilla', 352334),
    (200000, 4096, 'SIVF', 263014),
    (200000, 4096, 'Vanilla', 443137),
    (200000, 16384, 'SIVF', 205614),
    (200000, 16384, 'Vanilla', 320675),

    # --- 500k Vectors ---
    (500000, 1024, 'SIVF', 69784),
    (500000, 1024, 'Vanilla', 176131),
    (500000, 4096, 'SIVF', 117890),
    (500000, 4096, 'Vanilla', 227213),
    (500000, 16384, 'SIVF', 100102),
    (500000, 16384, 'Vanilla', 213205),
]

df = pd.DataFrame(data, columns=['NB', 'nlist', 'System', 'QPS'])
df['QPS_Thousands'] = df['QPS'] / 1000

# Calculate Efficiency Ratio (SIVF / Vanilla)
pivot_sivf = df[df['System'] == 'SIVF'].set_index(['NB', 'nlist'])['QPS']
pivot_vanilla = df[df['System'] == 'Vanilla'].set_index(['NB', 'nlist'])['QPS']
relative_df = (pivot_sivf / pivot_vanilla).reset_index()
relative_df.rename(columns={'QPS': 'Relative'}, inplace=True)

# ==========================================
# 2. Plotting Configuration (Horizontal 1x3, HUGE FONTS)
# ==========================================
sns.set_theme(style="whitegrid", font_scale=2.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42

# Wide figsize to maintain 1x3 aspect ratio suitable for column compression
fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

# Common styling parameters
lw = 5          # Thicker lines
ms = 16         # Larger markers
title_size = 24 # Huge titles
label_size = 22 # Huge labels
tick_size = 20  # Huge ticks

# -------------------------------------------------------
# Subplot 1: Scalability (Throughput vs Database Size)
# nlist=4096 is chosen as the representative configuration
# -------------------------------------------------------
subset_nb = df[df['nlist'] == 4096]
sns.lineplot(
    data=subset_nb, 
    x='NB', 
    y='QPS_Thousands', 
    hue='System', 
    style='System', 
    markers=True, 
    dashes=False, 
    linewidth=lw,
    markersize=ms,
    palette=['#d7191c', '#2b83ba'], # Red(SIVF) vs Blue(Vanilla)
    ax=axes[0]
)
axes[0].set_title('(a) Search Scalability', fontsize=title_size, weight='bold', pad=15)
axes[0].set_xlabel('Database Size', fontsize=label_size)
axes[0].set_ylabel('Search Throughput (10^3 QPS)', fontsize=label_size)

axes[0].set_xticks([100000, 200000, 500000])
axes[0].set_xticklabels(['100k', '200k', '500k'], fontsize=tick_size)
axes[0].tick_params(axis='y', labelsize=tick_size)
axes[0].set_ylim(0, 600) # Peak is approx 560k
axes[0].legend(title=None, loc='upper right', frameon=True, fontsize=tick_size, markerscale=2.0)

# -------------------------------------------------------
# Subplot 2: Impact of Granularity (Throughput vs nlist)
# Fixed at NB=500k
# -------------------------------------------------------
subset_nlist = df[df['NB'] == 500000]
sns.barplot(
    data=subset_nlist, 
    x='nlist', 
    y='QPS_Thousands', 
    hue='System', 
    palette=['#d7191c', '#2b83ba'],
    edgecolor='black',
    linewidth=2.5,
    ax=axes[1]
)
# Add bar labels
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.0f', padding=5, fontsize=tick_size, weight='bold')

axes[1].set_title('(b) Search Granularity', fontsize=title_size, weight='bold', pad=15)
axes[1].set_xlabel('Clusters (nlist)', fontsize=label_size)
axes[1].set_ylabel('') # Implicit y-axis
axes[1].tick_params(axis='x', labelsize=tick_size)
axes[1].tick_params(axis='y', labelsize=tick_size)
axes[1].set_ylim(0, 260)
axes[1].legend(title=None, loc='upper left', fontsize=tick_size)

# -------------------------------------------------------
# Subplot 3: Efficiency Ratio Heatmap
# Visualizes the performance gap (SIVF / Vanilla)
# -------------------------------------------------------
heatmap_data = relative_df.pivot(index="nlist", columns="NB", values="Relative")
annot_labels = heatmap_data.applymap(lambda v: f"{v:.2f}x")

sns.heatmap(
    heatmap_data, 
    annot=annot_labels, 
    fmt="",
    cmap="RdYlGn", # Red (low efficiency) to Green (high efficiency)
    linewidths=2.5, 
    linecolor='white',
    cbar_kws={'label': 'Efficiency (SIVF / Vanilla)'},
    annot_kws={"size": 22, "weight": "bold"},
    ax=axes[2]
)
# Adjust colorbar font
cbar = axes[2].collections[0].colorbar
cbar.ax.tick_params(labelsize=tick_size)
cbar.set_label('Efficiency (SIVF / Vanilla)', fontsize=label_size)

axes[2].set_title('(c) Efficiency Ratio', fontsize=title_size, weight='bold', pad=15)
axes[2].set_xlabel('Database Size', fontsize=label_size)
axes[2].set_ylabel('Clusters (nlist)', fontsize=label_size)
axes[2].set_xticklabels(['100k', '200k', '500k'], fontsize=tick_size)
axes[2].set_yticklabels(heatmap_data.index, rotation=0, fontsize=tick_size)

# ==========================================
# Save
# ==========================================
output_path = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/performance_search.pdf')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")