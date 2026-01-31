import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np, os

def plot_grid():
    # ================= DATA SETUP =================
    
# 1. Deep1B (96d) - [Sparsified: 10 points -> 6 points]
    # Removed intermediate points (97.44, 99.16, 99.45, 99.66) to reduce clutter
    deep_base_r = [43.63, 79.31, 89.24, 95.21]
    deep_base_q = [35613, 32271, 29093, 23733]
    
    deep_sivf_r = [43.65, 79.42, 89.25, 95.24]
    deep_sivf_q = [448640, 110107, 59929, 32318]

    # 2. SIFT1M (128d) - [Sparsified: 10 points -> 6 points]
    # Removed intermediate points (97.71, 99.54, 99.73, 99.84) to reduce clutter
    sift_base_r = [36.23, 74.31, 86.84, 94.85]
    sift_base_q = [34681, 30627, 26417, 20541]
    
    sift_sivf_r = [36.32, 74.19, 86.77, 94.81]
    sift_sivf_q = [323857, 75885, 40619, 21508]
    
    # 3. T2I-1B (200d)
    t2i_base_r  = [74.28, 83.05, 89.59, 94.03]
    t2i_base_q  = [8616, 8076, 7140, 5490]
    
    t2i_sivf_r  = [74.29, 83.08, 89.60, 94.05]
    t2i_sivf_q  = [69431, 37955, 20207, 10751]

    # 4. GIST1M (960d) - [UPDATED: Hybrid Pareto Frontier]
    # Strategy: 
    #   - Low Recall (30-60%): Use nlist=4096 (High QPS)
    #   - High Recall (65-95%): Use nlist=8192 (Better Recall ceiling)
    
    # Baseline: Points from nlist=4096 (nprobe 2-64) + nlist=8192 (nprobe 128)
    gist_base_r = [31.7, 43.8, 57.2, 70.2, 81.7, 91.8, 94.4]
    gist_base_q = [4765, 4953, 4675, 4366, 3784, 2938, 1776]

    # SIVF: Points from nlist=4096 (nprobe 2-8) + nlist=8192 (nprobe 16-128)
    gist_sivf_r = [32.1, 44.1, 59.8, 66.4, 77.5, 87.0, 95.2]
    gist_sivf_q = [57506, 30412, 14792, 14604, 7321, 3859, 1936]

    datasets = [
        {"name": "Deep1B (96D)", "br": deep_base_r, "bq": deep_base_q, "sr": deep_sivf_r, "sq": deep_sivf_q},
        {"name": "SIFT1M (128D)", "br": sift_base_r, "bq": sift_base_q, "sr": sift_sivf_r, "sq": sift_sivf_q},
        {"name": "T2I-1B (200D)", "br": t2i_base_r, "bq": t2i_base_q, "sr": t2i_sivf_r, "sq": t2i_sivf_q},
        {"name": "GIST1M (960D)", "br": gist_base_r, "bq": gist_base_q, "sr": gist_sivf_r, "sq": gist_sivf_q}
    ]

    # ================= PLOTTING CONFIG =================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['font.size'] = 14
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, data in enumerate(datasets):
        ax = axes[i]
        
        # Plot Baseline
        ax.plot(data["br"], data["bq"], 'o-', label='Baseline (GPU IVF)', 
                color='#1f77b4', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        
        # Plot SIVF
        ax.plot(data["sr"], data["sq"], 's-', label='SIVF (Ours)', 
                color='#d62728', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        
        # Titles
        ax.set_title(data["name"], fontsize=18, pad=10)
        
        # Labels
        if i >= 2:
            ax.set_xlabel('Recall@10 (%)', fontsize=16)
        if i % 2 == 0:
            ax.set_ylabel('QPS (log)', fontsize=16)
            
        # Log Scale
        ax.set_yscale('log')
        
        # Grid
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        # Ticks formatting
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Legend (Only in the first plot)
        if i == 0:
            ax.legend(fontsize=14, loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    output_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, 'pareto.pdf')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

if __name__ == "__main__":
    plot_grid()