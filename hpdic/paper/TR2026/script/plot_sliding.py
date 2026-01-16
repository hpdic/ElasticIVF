"""
plot_sliding.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization for Streaming Benchmark (SIFT1M + GIST1M).
Generates a dual-subplot figure comparing the sliding window latency 
of SIVF vs. Baseline across different dimensionalities using the latest
experimental data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. Experimental Data Input (From latest logs)
# ==========================================
steps = np.arange(0, 10)

# --- SIFT1M Data (128d, 200k Window, 10k Batch) ---
# Baseline Total Latency (ms)
sift_base = [
    657.33, 372.83, 354.71, 359.04, 354.78, 
    358.73, 358.71, 354.36, 357.50, 362.39
]
# SIVF Total Latency (ms)
sift_sivf = [
    2.65, 2.29, 2.26, 2.20, 2.22, 
    2.21, 2.15, 2.17, 2.17, 2.20
]

# --- GIST1M Data (960d, 100k Window, 5k Batch) ---
# Baseline Total Latency (ms)
gist_base = [
    1528.96, 1147.55, 1129.04, 1126.18, 1130.05, 
    1123.88, 1121.52, 1119.14, 1116.96, 1121.21
]
# SIVF Total Latency (ms)
gist_sivf = [
    4.58, 4.48, 4.57, 4.42, 4.51, 
    4.20, 4.11, 4.11, 4.14, 4.10
]

# Calculate Speedup for the stable phase (Steps 1-9), excluding Step 0 warmup
sift_speedup = np.mean(sift_base[1:]) / np.mean(sift_sivf[1:])
gist_speedup = np.mean(gist_base[1:]) / np.mean(gist_sivf[1:])

print(f"SIFT Average Speedup: {sift_speedup:.1f}x")
print(f"GIST Average Speedup: {gist_speedup:.1f}x")

# ==========================================
# 2. Plotting Configuration
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2

# --- Modification: Taller Aspect Ratio ---
# Changed from (12, 5) to (10, 6)
# Width reduced, Height increased -> "Taller and Narrower" look
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
plt.subplots_adjust(wspace=0.18, bottom=0.15) 

# Generic function to plot each subplot
def plot_subplot(ax, base_data, sivf_data, title, speedup_val):
    # Plot Baseline (Blue, Dashed)
    ax.plot(steps, base_data, label='Faiss Baseline (Roundtrip)', 
            marker='o', linestyle='--', linewidth=2, markersize=8, color='#1f77b4')
    
    # Plot SIVF (Orange, Solid)
    ax.plot(steps, sivf_data, label='SIVF (Native In-Place)', 
            marker='*', linestyle='-', linewidth=2, markersize=12, color='#ff7f0e')
    
    # Axis Settings
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel('Sliding Window Step', fontweight='bold')
    ax.set_xticks(steps)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    # Use Logarithmic Scale due to the significant performance gap
    ax.set_yscale('log')
    
    # Annotation: Speedup Text Box
    ax.text(4.5, np.mean(base_data)/4, 
            f"~{speedup_val:.0f}x Faster", 
            ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#d62728',
            bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.4', linewidth=1.5))
    
    # Annotation: Arrow indicating the drop in latency
    ax.annotate('', xy=(2, np.mean(sivf_data)*1.5), xytext=(2, np.mean(base_data)*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# ==========================================
# 3. Render Subplots
# ==========================================

# Subplot 1: SIFT1M Results
plot_subplot(ax1, sift_base, sift_sivf, '(a) SIFT1M (128D)', sift_speedup)
ax1.set_ylabel('Latency per Step (ms) [Log Scale]', fontweight='bold')
ax1.set_ylim(1, 1000) 
ax1.legend(loc='center right', fontsize=11, frameon=True)

# Subplot 2: GIST1M Results
plot_subplot(ax2, gist_base, gist_sivf, '(b) GIST1M (960D)', gist_speedup)

# --- Y-Axis on Right Side ---
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Latency per Step (ms) [Log Scale]', fontweight='bold')

ax2.set_ylim(1, 2000) 
ax2.legend(loc='center right', fontsize=11, frameon=True)

# ==========================================
# 4. Save Output
# ==========================================
output_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_path = os.path.join(output_dir, 'eval_sliding.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {save_path}")