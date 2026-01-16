"""
plot_sivf_add.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for SIVF ingestion throughput benchmark.
Generates a composite figure comparing SIVF vs. Vanilla Faiss across
different database sizes and cluster configurations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. Input Benchmark Data (2026-01-13: 1M/2M/4M)
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
# 2. Plotting Configuration (Wide Layout)
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['pdf.fonttype'] = 42 

# Create a 1x3 subplot layout
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# -------------------------------------------------------
# Subplot 1: Scalability (Throughput vs Database Size)
# Fix nlist=4096 (Most stable and representative configuration)
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
    linewidth=3,
    markersize=10,
    palette=['#d7191c', '#2b83ba'], # Red(SIVF) vs Blue(Vanilla)
    ax=axes[0]
)
axes[0].set_title('(a) Scalability with Data Size (nlist=4096)', fontsize=14, weight='bold', pad=12)
axes[0].set_xlabel('Database Size (Vectors)', fontsize=12)
axes[0].set_ylabel('Throughput (Million vec/s)', fontsize=12)

# Update ticks to 1M, 2M, 4M
axes[0].set_xticks([1000000, 2000000, 4000000])
axes[0].set_xticklabels(['1M', '2M', '4M'])
axes[0].set_ylim(0, 7) # SIVF peaks around 6.1M, setting limit to 7 for aesthetics
axes[0].legend(title=None, loc='center right', frameon=True)

# -------------------------------------------------------
# Subplot 2: Impact of Clustering (Throughput vs nlist)
# Fix NB=4M (Max stress test)
# -------------------------------------------------------
subset_nlist = df[df['NB'] == 4000000]
bar_plot = sns.barplot(
    data=subset_nlist, 
    x='nlist', 
    y='QPS_Millions', 
    hue='System', 
    palette=['#d7191c', '#2b83ba'],
    edgecolor='black',
    linewidth=1,
    ax=axes[1]
)
# Add value labels
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f', padding=3, fontsize=11)

axes[1].set_title('(b) Impact of Granularity (4M Vectors)', fontsize=14, weight='bold', pad=12)
axes[1].set_xlabel('Number of Clusters (nlist)', fontsize=12)
axes[1].set_ylabel('Throughput (Million vec/s)', fontsize=12)
axes[1].set_ylim(0, 7)
axes[1].legend(title=None, loc='upper right')

# -------------------------------------------------------
# Subplot 3: Speedup Heatmap (Summary)
# -------------------------------------------------------
heatmap_data = speedup_df.pivot(index="nlist", columns="NB", values="Speedup")

# Manually create an annotation string matrix
annot_labels = heatmap_data.applymap(lambda v: f"{v:.2f}x")

sns.heatmap(
    heatmap_data, 
    annot=annot_labels, 
    fmt="", 
    cmap="YlGnBu", 
    linewidths=1, 
    linecolor='white',
    cbar_kws={'label': 'Speedup Factor'},
    annot_kws={"size": 13, "weight": "bold"},
    ax=axes[2]
)
axes[2].set_title('(c) Speedup Factor (ElasticIVF vs Vanilla)', fontsize=14, weight='bold', pad=12)
axes[2].set_xlabel('Database Size (NB)', fontsize=12)
axes[2].set_ylabel('Number of Clusters (nlist)', fontsize=12)
axes[2].set_xticklabels(['1M', '2M', '4M'])
axes[2].set_yticklabels(heatmap_data.index, rotation=0)

# ==========================================
# Save
# ==========================================
# Save figure to file
plt.savefig(os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/performance_add.pdf'), format='pdf', dpi=300, bbox_inches='tight')