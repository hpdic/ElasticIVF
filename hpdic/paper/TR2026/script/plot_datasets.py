"""
plot_datasets.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for the Overall Evaluation Summary.
Generates three high-contrast bar charts comparing SIVF against the Faiss Baseline
across three standard datasets (SIFT1M, T2I-1B, GIST1M).
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
datasets = ['SIFT1M\n(128D)', 'T2I-1B\n(200D)', 'GIST1M\n(960D)']

# Ingestion Throughput (Vectors/sec)
add_base = [35901, 34596, 23492]
add_sivf = [3783727, 2908835, 852742]

# Deletion Latency (ms)
del_base = [1626.0, 2416.2, 11843.0]
del_sivf = [0.86, 0.87, 0.89]

# Search Throughput (Queries/sec)
search_base = [26702, 18635, 3640]
search_sivf = [40933, 17796, 1344]

# ==========================================
# 3. Core Plotting Function
# ==========================================

def draw_bar_chart(ylabel, data_base, data_sivf, filename_suffix, log_scale=False, mode="higher_better"):
    x = np.arange(len(datasets))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 8))
    
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
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 2000)
    else:
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 1.6)

    def autolabel(rects, is_sivf=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # Value Formatting
            if height >= 1000000: val_text = f'{height/1000000:.2f}M'
            elif height >= 1000: val_text = f'{height/1000:.0f}k'
            elif height < 10: val_text = f'{height:.2f}'
            else: val_text = f'{int(height)}'

            # Annotate raw value
            ax.annotate(val_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 10),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=22)
            
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

                offset = 45 if log_scale else 45
                
                ax.annotate(txt,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset), 
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            fontsize=24, color='black') # No Bold

    autolabel(rects1)
    autolabel(rects2, is_sivf=False)

    plt.tight_layout()
    
    full_path = os.path.join(OUTPUT_DIR, f"{filename_suffix}.pdf")
    plt.savefig(full_path, dpi=300)
    print(f"[Success] Generated: {full_path}")

# ==========================================
# 4. Generate Figures
# ==========================================

draw_bar_chart(
    ylabel='Throughput (vec/s)', 
    data_base=add_base,
    data_sivf=add_sivf,
    filename_suffix='eval_ingestion',
    log_scale=True,
    mode="higher_better"
)

draw_bar_chart(
    ylabel='Latency (ms)',
    data_base=del_base,
    data_sivf=del_sivf,
    filename_suffix='eval_deletion',
    log_scale=True,
    mode="lower_better"
)

draw_bar_chart(
    ylabel='Query Throughput (QPS)',
    data_base=search_base,
    data_sivf=search_sivf,
    filename_suffix='eval_search',
    log_scale=False,
    mode="higher_better"
)