"""
plot_datasets.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for the Overall Evaluation Summary.
Generates three high-contrast bar charts comparing SIVF against the Faiss Baseline
across two standard datasets (SIFT1M, GIST1M):
1. Ingestion Throughput (Log Scale)
2. Deletion Latency (Log Scale)
3. Search Throughput (Linear Scale)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 0. Output Configuration
# ==========================================
OUTPUT_DIR = os.path.expanduser("~/ElasticIVF/hpdic/paper/TR2026/figures")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. Font and Style Configuration (Large Fonts)
# ==========================================
# Use Serif fonts (Times New Roman) to match standard academic paper styles
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- Critical: Large Font Sizes for Legibility ---
plt.rcParams['font.size'] = 22          # Global base font size
plt.rcParams['axes.labelsize'] = 26     # Axis labels (X/Y)
plt.rcParams['axes.titlesize'] = 26     # Figure titles
plt.rcParams['xtick.labelsize'] = 22    # X-tick labels
plt.rcParams['ytick.labelsize'] = 22    # Y-tick labels
plt.rcParams['legend.fontsize'] = 20    # Legend text

# Color Palette (High Contrast / Print Friendly)
# Muted Blue for Baseline, Safety Orange for SIVF to highlight the proposed method
COLOR_BASE = '#1f77b4' 
COLOR_SIVF = '#ff7f0e' 

# ==========================================
# 2. Data Preparation
# ==========================================
datasets = ['SIFT1M (128D)', 'GIST1M (960D)']

# Ingestion Throughput (Vectors/sec) - Higher is Better
add_base = [35901, 23492]
add_sivf = [3783727, 852742]

# Deletion Latency (ms) - Lower is Better
del_base = [1626.0, 11843.0]
del_sivf = [0.86, 0.89]

# Search Throughput (Queries/sec) - Higher is Better
search_base = [26702, 3640]
search_sivf = [40933, 1344]

# ==========================================
# 3. Core Plotting Function
# ==========================================

def draw_bar_chart(ylabel, data_base, data_sivf, filename_suffix, log_scale=False, mode="higher_better"):
    """
    Generates and saves a comparative bar chart.
    
    Args:
        ylabel (str): Label for the Y-axis.
        data_base (list): Metrics for Faiss Baseline.
        data_sivf (list): Metrics for SIVF.
        filename_suffix (str): Suffix for the output PDF file.
        log_scale (bool): Whether to use a logarithmic scale for the Y-axis.
        mode (str): "higher_better" or "lower_better" for calculating speedup annotations.
    """
    x = np.arange(len(datasets))
    width = 0.35  

    # Use a slightly larger canvas (9x7) to accommodate large fonts
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Plot Bars: Add hatching '//' to Baseline for black-and-white distinction
    rects1 = ax.bar(x - width/2, data_base, width, label='Faiss Baseline', 
                    color=COLOR_BASE, alpha=0.7, edgecolor='black', hatch='//')
    rects2 = ax.bar(x + width/2, data_sivf, width, label='SIVF (Ours)', 
                    color=COLOR_SIVF, alpha=0.9, edgecolor='black')

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.legend(frameon=False) # Remove legend box for a cleaner look
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Adjust Y-axis limits to leave room for top annotations
    if log_scale:
        ax.set_yscale('log')
        # Log scale: set top limit higher (50x) to prevent text clipping
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 50)
    else:
        # Linear scale: set top limit to 1.4x max value
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 1.4)

    # Helper to annotate values and speedups on bars
    def autolabel(rects, is_sivf=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # --- Value Formatting ---
            if height >= 1000000:
                val_text = f'{height/1000000:.2f}M'
            elif height >= 1000:
                val_text = f'{height/1000:.0f}k' # Drop decimals for compactness
            elif height < 10:
                val_text = f'{height:.2f}'
            else:
                val_text = f'{int(height)}'

            # Annotate raw value (Font size 18)
            ax.annotate(val_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18)
            
            # --- Speedup Factor Annotation (Only on SIVF bars) ---
            if is_sivf:
                base = data_base[i]
                curr = data_sivf[i]
                
                if mode == "lower_better": # Latency: Base / Curr
                    speedup = base / curr
                    if speedup > 1000:
                        txt = f"{speedup/1000:.1f}k x" # e.g., 13k x
                    else:
                        txt = f"{speedup:.0f}x"
                else: # Throughput: Curr / Base
                    speedup = curr / base
                    if speedup < 1:
                        txt = f"{speedup:.2f}x"
                    else:
                        txt = f"{speedup:.0f}x"

                # Annotate speedup (Font size 20, Bold)
                offset = 25 if log_scale else 30
                if mode == "lower_better": offset = 25

                ax.annotate(txt,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset), 
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            fontsize=20, fontweight='bold', color='black')

    autolabel(rects1)
    autolabel(rects2, is_sivf=True)

    plt.tight_layout()
    
    full_path = os.path.join(OUTPUT_DIR, f"{filename_suffix}.pdf")
    plt.savefig(full_path, dpi=300)
    print(f"[Success] Generated: {full_path}")

# ==========================================
# 4. Generate Figures
# ==========================================

# Figure 1: Ingestion Throughput 
draw_bar_chart(
    ylabel='Throughput (vecs/s)', 
    data_base=add_base,
    data_sivf=add_sivf,
    filename_suffix='eval_ingestion',
    log_scale=True,
    mode="higher_better"
)

# Figure 2: Deletion Latency 
draw_bar_chart(
    ylabel='Latency (ms)',
    data_base=del_base,
    data_sivf=del_sivf,
    filename_suffix='eval_deletion',
    log_scale=True,
    mode="lower_better"
)

# Figure 3: Search Performance 
draw_bar_chart(
    ylabel='Query Throughput (QPS)',
    data_base=search_base,
    data_sivf=search_sivf,
    filename_suffix='eval_search',
    log_scale=False,
    mode="higher_better"
)