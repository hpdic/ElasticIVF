"""
plot_sivf_delete.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for SIVF deletion performance benchmark.
Generates a dual axis bar chart comparing the deletion latency and throughput
between SIVF and the Faiss Baseline on a batch of 10k vectors.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. Data Preparation (From 2026-01-13 Benchmark)
# ==========================================
# Baseline (Faiss): Consistent latency around 202.2 ms for 10k vectors
raw_baseline = [202.2, 202.2, 202.2] 

# SIVF (Proposed): Latency measurements from three independent runs
# Values: 0.681ms, 0.683ms, 0.668ms
raw_sivf = [0.681474, 0.683111, 0.667850]

data_groups = [raw_baseline, raw_sivf]
methods = ['Faiss IVF\n(Baseline)', 'SIVF\n(Ours)']

lat_means = [np.mean(g) for g in data_groups]
lat_stds = [np.std(g) for g in data_groups]

# Calculate Speedup Factor
speedup_val = lat_means[0] / lat_means[1]  # Approximately 298x

# Throughput calculation (vectors per second)
# Formula: 10000 vectors / (time_ms / 1000.0)
tp_groups = [[10000 / (x / 1000.0) for x in g] for g in data_groups]
tp_means = [np.mean(g) for g in tp_groups]
tp_stds = [np.std(g) for g in tp_groups]

# ==========================================
# 2. Plotting Configuration (Publication Quality)
# ==========================================
# Use Type 42 fonts (TrueType) to ensure text is editable in PDF viewers
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif' # Sans serif fonts are preferred for clarity
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Output directory configuration
save_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/') 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create figure: Wide layout (9x4.5) to fit double column paper format
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
plt.subplots_adjust(wspace=0.25, bottom=0.15, top=0.88, left=0.1, right=0.9)

colors = ['#4c72b0', '#c44e52'] # Palette: Blue vs Red

# ==========================================
# Subplot 1: Latency (Log Scale)
# ==========================================
bars1 = ax1.bar(methods, lat_means, yerr=lat_stds, capsize=5, 
               color=colors, width=0.5, edgecolor='black', linewidth=1,
               error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

ax1.set_yscale('log')
ax1.set_ylabel('Deletion Latency (ms)', fontweight='bold')
ax1.set_title('(a) Latency (Lower is Better)', fontweight='bold', pad=10)

# Set Y limit to accommodate annotations (Range: 0.1ms to 1500ms)
ax1.set_ylim(0.1, 1500) 

# Annotation: Baseline Value
base_bar = bars1[0]
ax1.text(base_bar.get_x() + base_bar.get_width()/2., base_bar.get_height() * 1.1, 
         f'{lat_means[0]:.1f} ms', ha='center', va='bottom', fontsize=10)

# Annotation: SIVF Value
sivf_bar = bars1[1]
h_sivf = sivf_bar.get_height()
ax1.text(sivf_bar.get_x() + sivf_bar.get_width()/2., h_sivf * 1.15, 
         f'{h_sivf:.3f} ms', ha='center', va='bottom', fontweight='bold', fontsize=11, color='#c44e52')

# Annotation: Speedup Arrow
# Draws an arrow indicating the reduction from Baseline to SIVF
ax1.annotate(f'{speedup_val:.0f}x Faster', 
             xy=(1.0, h_sivf * 1.5), 
             xytext=(0.6, lat_means[0] * 0.5), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold', ha='center', color='black')

# ==========================================
# Subplot 2: Throughput (Log Scale)
# ==========================================
bars2 = ax2.bar(methods, tp_means, yerr=tp_stds, capsize=5, 
                color=colors, width=0.5, edgecolor='black', linewidth=1,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

ax2.set_yscale('log')
ax2.set_ylabel('Throughput (vecs/sec)', fontweight='bold')
ax2.set_title('(b) Throughput (Higher is Better)', fontweight='bold', pad=10)

# Move Y axis label to the right side for better layout balance
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")

# Adjust limits for throughput range
ax2.set_ylim(1e4, 1e8) 

# Annotation: Value Labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.2,
            f'{height:.1e}',
            ha='center', va='bottom', fontsize=10)

# ==========================================
# Final Save
# ==========================================
save_path = os.path.join(save_dir, 'performance_delete.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Figure saved to: {save_path}")
plt.show()