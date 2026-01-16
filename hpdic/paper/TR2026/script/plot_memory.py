"""
plot_memory_scalability.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization of Memory Scalability.
Demonstrates that SIVF memory usage grows linearly and closely tracks the
theoretical compact baseline, using a grouped bar chart comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# ==========================================
# 1. Experimental Data (From your logs)
# ==========================================
# X-Axis: Vector Counts
counts = np.array([100000, 200000, 500000, 1000000])
counts_labels = ['100K', '200K', '500K', '1M']

# --- SIFT1M (128D) ---
sift_base_mb = np.array([49.59, 99.18, 247.96, 495.91])
sift_sivf_mb = np.array([49.97, 99.95, 249.86, 499.73])

# --- GIST1M (960D) ---
gist_base_mb = np.array([366.97, 733.95, 1834.87, 3669.74])
gist_sivf_mb = np.array([367.36, 734.71, 1836.78, 3673.55])

# ==========================================
# 2. Plotting Configuration
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2

# --- Modification 1: Wider Aspect Ratio ---
# Changed from (8, 6) back to wider (10, 5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# Adjust wspace to make room for the right y-axis label
plt.subplots_adjust(wspace=0.25, bottom=0.15, right=0.88)

# Helper for formating Y-axis as MB/GB
def mb_formatter(x, pos):
    if x >= 1000:
        return f'{x/1000:.1f} GB'
    return f'{int(x)} MB'

def plot_growth_bar(ax, base, sivf, title, overhead_txt):
    # Bar configuration
    width = 0.35
    ind = np.arange(len(counts)) # X-axis positions for groups

    # Plot Baseline Bars (Left, Hatched)
    ax.bar(ind - width/2, base, width, label='Compact Baseline',
           color='white', edgecolor='black', hatch='//', alpha=0.8)

    # Plot SIVF Bars (Right, Solid Orange)
    ax.bar(ind + width/2, sivf, width, label='SIVF (Ours)',
           color='#ff7f0e', edgecolor='black', alpha=0.9)

    # Settings
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Vectors', fontweight='bold')
    
    # X-ticks and labels
    ax.set_xticks(ind)
    ax.set_xticklabels(counts_labels)
    
    # Y-axis formatter
    ax.yaxis.set_major_formatter(FuncFormatter(mb_formatter))
    
    # Grid only on Y-axis for bar charts
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(fontsize=11, frameon=True, loc='upper left')
    
    # Place it between the 2nd and 3rd group (x=1.5), near the top (y=90% max)
    # Using va='top' to anchor it down from that point
    ax.text(0.9, ax.get_ylim()[1] * 0.7, 
            overhead_txt, 
            color='#d62728', fontweight='bold', ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='#d62728', alpha=0.95, boxstyle='round,pad=0.4'))

# ==========================================
# 3. Render Subplots
# ==========================================

# --- Left Plot: SIFT1M ---
plot_growth_bar(ax1, sift_base_mb, sift_sivf_mb, '(a) SIFT1M Memory Growth', "Stable 0.77% Overhead")
ax1.set_ylabel('VRAM Usage', fontweight='bold')

# --- Right Plot: GIST1M ---
plot_growth_bar(ax2, gist_base_mb, gist_sivf_mb, '(b) GIST1M Memory Growth', "Stable 0.10% Overhead")

# Move Y-Axis to Right Side
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('VRAM Usage', fontweight='bold')

# ==========================================
# 4. Save
# ==========================================
output_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_path = os.path.join(output_dir, 'eval_memory.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {save_path}")