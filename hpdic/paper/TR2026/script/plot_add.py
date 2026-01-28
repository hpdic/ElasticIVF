"""
plot_sivf_add.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for SIVF ingestion throughput benchmark.
Generates a 1x3 composite figure (Horizontal) with OVERSIZED fonts
to survive extreme scaling (e.g., fitting 3 plots into a single column width).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. Input Benchmark Data
# ==========================================
data = [
    # NB, nlist, System, QPS
    # --- 1M Vectors ---
    (1000000, 1024, 'SIVF', 5728030),
    (1000000, 1024, 'Vanilla', 2077930),
    (1000000, 2048, 'SIVF', 5307344),
    (1000000, 2048, 'Vanilla', 1863412),
    (1000000, 4096, 'SIVF', 4182976),
    (1000000, 4096, 'Vanilla', 1667996),
    
    # --- 2M Vectors ---
    (2000000, 1024, 'SIVF', 6095778),
    (2000000, 1024, 'Vanilla', 2115055),
    (2000000, 2048, 'SIVF', 3574601),
    (2000000, 2048, 'Vanilla', 1709697),
    (2000000, 4096, 'SIVF', 4178445),
    (2000000, 4096, 'Vanilla', 1742637),

    # --- 4M Vectors ---
    (4000000, 1024, 'SIVF', 4007265),
    (4000000, 1024, 'Vanilla', 2237242),
    (4000000, 2048, 'SIVF', 5293120),
    (4000000, 2048, 'Vanilla', 2029887),
    (4000000, 4096, 'SIVF', 4211634),
    (4000000, 4096, 'Vanilla', 1696161),
]

df = pd.DataFrame(data, columns=['NB', 'nlist', 'System', 'QPS'])
df['QPS_Millions'] = df['QPS'] / 1e6

# Calculate Speedup for Heatmap
pivot_sivf = df[df['System'] == 'SIVF'].set_index(['NB', 'nlist'])['QPS']
pivot_vanilla = df[df['System'] == 'Vanilla'].set_index(['NB', 'nlist'])['QPS']
speedup_df = (pivot_sivf / pivot_vanilla).reset_index()
speedup_df.rename(columns={'QPS': 'Speedup'}, inplace=True)

# ==========================================
# 2. Plotting Configuration (Horizontal 1x3, HUGE FONTS)
# ==========================================
# Use a very large font scale so text remains readable when image is shrunk
sns.set_theme(style="whitegrid", font_scale=2.2) 
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['pdf.fonttype'] = 42 

# Wide figsize to maintain 1x3 aspect ratio.
# When you insert this into a column, LaTeX will shrink the 24-inch width to ~3.5 inches.
# The fonts need to be massive to compensate (2.2x scale).
fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)

# Common params for visibility
lw = 5          # Thicker lines
ms = 16         # Larger markers
title_size = 24 # Huge titles
label_size = 22 # Huge labels
tick_size = 20  # Huge ticks

# -------------------------------------------------------
# Subplot 1: Scalability (Throughput vs Database Size)
# Fix nlist=4096
# -------------------------------------------------------
subset_nb = df[df['nlist'] == 4096]
sns.lineplot(
    data=subset_nb, 
    x='NB', 
    y='QPS_Millions', 
    hue='System', 
    style='System', 
    markers=True, 
    dashes=False, 
    linewidth=lw,
    markersize=ms,
    palette=['#d7191c', '#2b83ba'], 
    ax=axes[0]
)
axes[0].set_title('(a) Scalability', fontsize=title_size, weight='bold', pad=15)
axes[0].set_xlabel('Database Size', fontsize=label_size)
axes[0].set_ylabel('Throughput (M vec/s)', fontsize=label_size)

axes[0].set_xticks([1000000, 2000000, 4000000])
axes[0].set_xticklabels(['1M', '2M', '4M'], fontsize=tick_size)
axes[0].tick_params(axis='y', labelsize=tick_size)
axes[0].set_ylim(0, 7)
# Legend needs to be huge too
axes[0].legend(title=None, loc='center right', frameon=True, fontsize=tick_size, markerscale=2.0)

# -------------------------------------------------------
# Subplot 2: Impact of Clustering (Throughput vs nlist)
# Fix NB=4M
# -------------------------------------------------------
subset_nlist = df[df['NB'] == 4000000]
bar_plot = sns.barplot(
    data=subset_nlist, 
    x='nlist', 
    y='QPS_Millions', 
    hue='System', 
    palette=['#d7191c', '#2b83ba'],
    edgecolor='black',
    linewidth=2.5,
    ax=axes[1]
)
# Add massive value labels
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f', padding=5, fontsize=tick_size, weight='bold')

axes[1].set_title('(b) Granularity', fontsize=title_size, weight='bold', pad=15)
axes[1].set_xlabel('Clusters (nlist)', fontsize=label_size)
axes[1].set_ylabel('') # Save space, y-axis shared implicitly by context
axes[1].tick_params(axis='x', labelsize=tick_size)
axes[1].tick_params(axis='y', labelsize=tick_size)
axes[1].set_ylim(0, 7)
axes[1].legend(title=None, loc='upper right', fontsize=tick_size)

# -------------------------------------------------------
# Subplot 3: Speedup Heatmap (Summary)
# -------------------------------------------------------
heatmap_data = speedup_df.pivot(index="nlist", columns="NB", values="Speedup")
annot_labels = heatmap_data.applymap(lambda v: f"{v:.2f}x")

sns.heatmap(
    heatmap_data, 
    annot=annot_labels, 
    fmt="", 
    cmap="YlGnBu", 
    linewidths=2.5, 
    linecolor='white',
    cbar_kws={'label': 'Speedup Factor'},
    annot_kws={"size": 22, "weight": "bold"}, # Huge numbers inside heatmap
    ax=axes[2]
)
# Adjust colorbar font
cbar = axes[2].collections[0].colorbar
cbar.ax.tick_params(labelsize=tick_size)
cbar.set_label('Speedup Factor', fontsize=label_size)

axes[2].set_title('(c) Speedup Factor', fontsize=title_size, weight='bold', pad=15)
axes[2].set_xlabel('Database Size', fontsize=label_size)
axes[2].set_ylabel('Clusters (nlist)', fontsize=label_size)
axes[2].set_xticklabels(['1M', '2M', '4M'], fontsize=tick_size)
axes[2].set_yticklabels(heatmap_data.index, rotation=0, fontsize=tick_size)

# ==========================================
# Save
# ==========================================
output_path = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/performance_add.pdf')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")