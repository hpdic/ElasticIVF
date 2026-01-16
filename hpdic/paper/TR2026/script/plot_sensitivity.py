"""
plot_sivf_sensitivity.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for SIVF parameter sensitivity analysis.
Plots Insertion vs. Deletion metrics (throughput and latency) across varying
memory pre-allocation factors (mv), slab redundancy factors (sl), and 
deletion batch sizes (b).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 1. Experimental Data (Benchmark: 2026-01-14)
# ============================================================
# Extracted from output logs. The trailing 1000 is the actual count of deleted vectors.
# Format: (mv, sl, b, add_qps, del_qps, add_sec, del_sec)
rows = [
    (1.10, 1.00, 1024,  2712883.37,   1077536.30,   0.003686,   0.000928),
    (1.10, 1.00, 8192,  2884060.18,   1584778.52,   0.003467,   0.000631),
    (1.10, 1.30, 1024,  2907062.97,   1630217.79,   0.003440,   0.000613),
    (1.10, 1.30, 8192,  2926934.35,   1687692.50,   0.003417,   0.000593),
    (1.50, 1.00, 1024,  2948788.71,   1708146.12,   0.003391,   0.000585),
    (1.50, 1.00, 8192,  2944191.67,   1675771.72,   0.003397,   0.000597),
    (1.50, 1.30, 1024,  2939841.43,   1693227.86,   0.003402,   0.000591),
    (1.50, 1.30, 8192,  3233038.04,   1655714.18,   0.003093,   0.000604),
]

mv_vals = [r[0] for r in rows]
sl_vals = [r[1] for r in rows]
b_vals  = [r[2] for r in rows]
add_qps = np.array([r[3] for r in rows]) / 1e6 
del_qps = np.array([r[4] for r in rows]) / 1e6 
add_ms  = np.array([r[5] for r in rows]) * 1000.0
del_ms  = np.array([r[6] for r in rows]) * 1000.0

# Construct X-axis labels: mv (MaxVec Factor), sl (Slab Factor), b (Batch Size)
x_labels = [f"mv:{m:.1f}\nsl:{s:.1f}\nb:{b}" for m, s, b in zip(mv_vals, sl_vals, b_vals)]

# ============================================================
# 2. Plotting Configuration
# ============================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

save_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

color_insert = '#4c72b0' 
color_delete = '#c44e52' 

def annotate_bar(ax, bars, fmt, color='#333333'):
    """Helper to add value annotations on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                fmt.format(height), ha='center', va='bottom', 
                fontsize=8.5, fontweight='bold', color=color, clip_on=False)

def plot_sensitivity(y1, y2, ylabel, title1, title2, out_name, fmt):
    """Generates a side-by-side bar chart for Insertion (a) vs Deletion (b)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    plt.subplots_adjust(wspace=0.2, bottom=0.22)
    
    x = np.arange(len(x_labels))
    
    # Subplot (a): Insertion
    bars1 = ax1.bar(x, y1, color=color_insert, edgecolor='black', linewidth=0.8, alpha=0.9)
    ax1.set_ylabel(ylabel, fontweight='bold')
    ax1.set_title(title1, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=7.5)
    ax1.set_ylim(0, max(y1) * 1.35) 
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    annotate_bar(ax1, bars1, fmt, color_insert)

    # Subplot (b): Deletion (Y-axis on the right)
    bars2 = ax2.bar(x, y2, color=color_delete, edgecolor='black', linewidth=0.8, alpha=0.9)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(ylabel, fontweight='bold')
    ax2.set_title(title2, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=7.5)
    ax2.set_ylim(0, max(y2) * 1.35)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    annotate_bar(ax2, bars2, fmt, color_delete)

    save_path = os.path.join(save_dir, f'{out_name}.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.show()

# ============================================================
# 3. Execution
# ============================================================

# Throughput Comparison
plot_sensitivity(add_qps, del_qps, "Throughput (M vec/s)", 
                 "(a) Insertion Throughput", "(b) Deletion Throughput",
                 "sivf_sensitivity_qps", "{:.2f}")

# Latency Comparison
plot_sensitivity(add_ms, del_ms, "Latency (ms)", 
                 "(a) Insertion Latency", "(b) Deletion Latency",
                 "sivf_sensitivity_latency", "{:.2f}")