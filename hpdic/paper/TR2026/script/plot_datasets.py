"""
plot_datasets.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for the Overall Evaluation Summary.
Generates three high-contrast bar charts comparing SIVF against the Faiss Baseline
across FOUR standard datasets (Deep1B, SIFT1M, T2I-1B, GIST1M).
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
# 1. Font and Style Configuration
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- Large Font Sizes ---
plt.rcParams['font.size'] = 24          
plt.rcParams['axes.labelsize'] = 28     
plt.rcParams['axes.titlesize'] = 28     
plt.rcParams['xtick.labelsize'] = 24    
plt.rcParams['ytick.labelsize'] = 24    
plt.rcParams['legend.fontsize'] = 22    

COLOR_BASE = '#1f77b4' 
COLOR_SIVF = '#ff7f0e' 

# ==========================================
# 2. Data Preparation
# ==========================================
# Datasets ordered by dimension: Deep1B (96D) -> SIFT1M (128D) -> T2I (200D) -> GIST (960D)
datasets = ['Deep1B\n(96D)', 'SIFT1M\n(128D)', 'T2I-1B\n(200D)', 'GIST1M\n(960D)']

# Ingestion Throughput (Vectors/sec)
add_base = [36375,   35901,   34596,   23492]
add_sivf = [4381030, 3783727, 2908835, 852742]

# Deletion Latency (ms)
del_base = [1182.0, 1626.0, 2416.2, 11843.0]
del_sivf = [0.86,   0.86,   0.87,   0.89]

# Search Throughput (Queries/sec)
search_base = [28913, 26702, 18635, 1776]
search_sivf = [59787, 40933, 17796, 1936]

# ==========================================
# 3. Core Plotting Function
# ==========================================

def draw_bar_chart(ylabel, data_base, data_sivf, filename_suffix, log_scale=False, mode="higher_better", ylim_factor=None):
    x = np.arange(len(datasets))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(9, 6)) # Adjusted size
    
    rects1 = ax.bar(x - width/2, data_base, width, label='Faiss Baseline', 
                    color=COLOR_BASE, alpha=0.7, edgecolor='black', hatch='//')
    rects2 = ax.bar(x + width/2, data_sivf, width, label='SIVF (Ours)', 
                    color=COLOR_SIVF, alpha=0.9, edgecolor='black')

    ax.set_ylabel(ylabel) 
    ax.set_xticks(x)
    ax.set_xticklabels(datasets) 
    ax.legend(frameon=False) 
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Adjust Y-axis limits
    if log_scale:
        ax.set_yscale('log')
        # Default factor 2000 is good for huge gaps (Add/Delete), 
        # but for Search (smaller gap), a smaller factor (e.g. 50) avoids too much empty space
        factor = ylim_factor if ylim_factor else 2000 
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * factor)
    else:
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 1.6)

    def autolabel(rects, is_sivf=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # Value Formatting (UPDATED: More precision for K)
            if height >= 1000000: val_text = f'{height/1000000:.2f}M'
            elif height >= 1000: val_text = f'{height/1000:.1f}K'
            elif height < 10: val_text = f'{height:.2f}'
            else: val_text = f'{int(height)}'

            # Annotate raw value
            ax.annotate(val_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 8),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18) 
            
            # Speedup Factor Annotation
            if is_sivf:
                base = data_base[i]
                curr = data_sivf[i]
                
                if mode == "lower_better": 
                    speedup = base / curr
                    if speedup > 1000: txt = f"{speedup/1000:.1f}k x" 
                    else: txt = f"{speedup:.0f}x"
                else: 
                    speedup = curr / base
                    if speedup < 1: txt = f"{speedup:.2f}x"
                    else: txt = f"{speedup:.0f}x"

                # Log scale needs different offset handling usually, but simple offset works if ylim is high enough
                offset = 35 
                
                ax.annotate(txt,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset), 
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            fontsize=20, color='black', weight='bold')

    autolabel(rects1)
    autolabel(rects2, is_sivf=False)

    plt.tight_layout()
    
    full_path = os.path.join(OUTPUT_DIR, f"{filename_suffix}.pdf")
    plt.savefig(full_path, dpi=300)
    print(f"[Success] Generated: {full_path}")

# ==========================================
# 4. Generate Figures
# ==========================================

# 1. Ingestion Throughput (Log Scale)
draw_bar_chart(
    ylabel='Throughput (vec/s)', 
    data_base=add_base,
    data_sivf=add_sivf,
    filename_suffix='eval_ingestion',
    log_scale=True,
    mode="higher_better"
)

# 2. Deletion Latency (Log Scale)
draw_bar_chart(
    ylabel='Latency (ms)',
    data_base=del_base,
    data_sivf=del_sivf,
    filename_suffix='eval_deletion',
    log_scale=True,
    mode="lower_better"
)

# 3. Search Throughput (Log Scale - MODIFIED)
# search gap is smaller (~2x), so we use a smaller ylim_factor (50) to avoid too much whitespace
draw_bar_chart(
    ylabel='Query Throughput (QPS)',
    data_base=search_base,
    data_sivf=search_sivf,
    filename_suffix='eval_search',
    log_scale=True,      
    mode="higher_better",
    ylim_factor=50       
)