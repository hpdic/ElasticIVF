import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np, os

def plot_grid():
    # ================= DATA SETUP =================
    
    # 1. Deep1B (96d)
    deep_base_r = [43.63, 79.31, 89.24, 95.21, 97.44, 98.18, 99.16, 99.45, 99.66, 99.80]
    deep_base_q = [35613, 32271, 29093, 23733, 19796, 18032, 13540, 11696, 9387, 7588]
    deep_sivf_r = [43.65, 79.42, 89.25, 95.24, 97.49, 98.19, 99.18, 99.45, 99.65, 99.80]
    deep_sivf_q = [448640, 110107, 59929, 32318, 21165, 17272, 11236, 9161, 7451, 5934]

    # 2. SIFT1M (128d)
    sift_base_r = [36.23, 74.31, 86.84, 94.85, 97.71, 98.55, 99.54, 99.73, 99.84, 99.90]
    sift_base_q = [34681, 30627, 26417, 20541, 16280, 14491, 10542, 9183, 7195, 5885]
    sift_sivf_r = [36.32, 74.19, 86.77, 94.81, 97.74, 98.59, 99.57, 99.74, 99.85, 99.91]
    sift_sivf_q = [323857, 75885, 40619, 21508, 13860, 11250, 7244, 5892, 4766, 3795]

    # 3. T2I-1B (200d)
    t2i_base_r  = [31.15, 61.01, 72.25, 81.49, 86.30, 88.26, 91.69, 93.01, 94.25, 95.41]
    t2i_base_q  = [33399, 28651, 24346, 18553, 14467, 12534, 9070, 7654, 6070, 4931]
    t2i_sivf_r  = [31.68, 61.41, 72.54, 81.67, 86.47, 88.41, 91.79, 93.09, 94.33, 95.48]
    t2i_sivf_q  = [269092, 62524, 33439, 17782, 11486, 9354, 6005, 4855, 3917, 3083]

    # 4. GIST1M (960d)
    gist_base_r = [23.51, 56.63, 72.06, 85.85, 92.42, 94.46, 97.77, 98.77, 99.41, 99.72]
    gist_base_q = [17210, 9000, 6094, 3612, 2478, 2007, 1339, 1107, 899, 722]
    gist_sivf_r = [23.36, 56.35, 72.28, 86.05, 92.46, 94.66, 97.77, 98.69, 99.31, 99.69]
    gist_sivf_q = [23315, 5132, 2642, 1336, 841, 671, 442, 353, 289, 230]

    datasets = [
        {"name": "Deep1B (96d)", "br": deep_base_r, "bq": deep_base_q, "sr": deep_sivf_r, "sq": deep_sivf_q},
        {"name": "SIFT1M (128d)", "br": sift_base_r, "bq": sift_base_q, "sr": sift_sivf_r, "sq": sift_sivf_q},
        {"name": "T2I-1M (200d)", "br": t2i_base_r, "bq": t2i_base_q, "sr": t2i_sivf_r, "sq": t2i_sivf_q},
        {"name": "GIST1M (960d)", "br": gist_base_r, "bq": gist_base_q, "sr": gist_sivf_r, "sq": gist_sivf_q}
    ]

    # ================= PLOTTING CONFIG =================
    # Serif font, No Bold
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['font.size'] = 14
    
    # 2x2 Grid
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
        
        # Titles & Labels
        ax.set_title(data["name"], fontsize=18, pad=10)
        
        # Only set Y-label for the left column, X-label for bottom row to save space (optional, but requested layout needs clarity)
        # Actually for a 2x2 grid, it's safer to label all or just outer ones. 
        # Let's label all but keep it clean.
        if i >= 2:
            ax.set_xlabel('Recall@10 (%)', fontsize=16)
        if i % 2 == 0:
            ax.set_ylabel('QPS (log)', fontsize=16)
            
        # Log Scale is CRITICAL here
        ax.set_yscale('log')
        
        # Grid
        ax.grid(True, which="major", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        # Ticks formatting
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Legend (Only in the first plot to avoid clutter, or all if space permits)
        # Given the curves move from top-left to bottom-right, legend usually fits in Top-Right or Bottom-Left.
        # Let's put it in the first plot only? Or all? 
        # Let's try putting it in Deep1B (Top-Left) and maybe GIST (Bottom-Right) or just once globally.
        # For papers, usually one legend is enough if styles are consistent.
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