"""
plot_mpi_gpu.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization for Multi-GPU Scalability (Weak Scaling).
Plots Insertion, Deletion, and Search throughput on a single Log-Scale chart.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# ============================================================
# 1. Experimental Data (Summary)
# ============================================================
gpus = [1, 2, 4]

# Data unit: Million QPS (M QPS)
# Source: Previous MPI Benchmarks
data_insert = [4.425, 11.027, 17.749]       # 17.75 M
data_delete = [13.908, 34.050, 64.006]      # 64.01 M
data_search = [0.005851, 0.011636, 0.023344] # 23.3 k -> 0.023 M

# ============================================================
# 2. Plotting Configuration
# ============================================================
# Optimize for IEEE/ACM Single Column width (~3.5 inches)
plt.rcParams['figure.figsize'] = (3.5, 2.2) 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3

save_dir = os.path.expanduser('~/hpdic/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Colors (Seaborn-like muted palette)
c_insert = '#1f77b4'  # Blue
c_delete = '#d62728'  # Red
c_search = '#2ca02c'  # Green

def plot_mpi_gpu():
    fig, ax = plt.subplots()
    
    # Plot Lines
    # Insert
    ax.plot(gpus, data_insert, marker='o', markersize=4, 
            label='Insert', color=c_insert, linewidth=1.5, linestyle='-')
    
    # Delete
    ax.plot(gpus, data_delete, marker='^', markersize=4, 
            label='Delete', color=c_delete, linewidth=1.5, linestyle='-')
    
    # Search
    ax.plot(gpus, data_search, marker='s', markersize=4, 
            label='Search', color=c_search, linewidth=1.5, linestyle='-')

    # Axes Setup
    ax.set_yscale('log') # Log scale is essential for orders of magnitude diff
    ax.set_ylabel('Throughput (M QPS)', fontweight='bold')
    ax.set_xlabel('# GPUs (V100)', fontweight='bold')
    
    # X-Axis Ticks
    ax.set_xticks(gpus)
    ax.set_xticklabels([str(g) for g in gpus])
    
    # Grid
    ax.grid(True, which="major", axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    
    # Legend
    # Place legend inside to save space, usually 'best' works or 'upper left'
    ax.legend(frameon=True, fontsize=7, loc='best', 
              fancybox=False, edgecolor='black', framealpha=0.8)

    # Tight Layout
    plt.tight_layout(pad=0.5)
    
    # Save
    save_path = os.path.join(save_dir, 'mpi_gpu.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    plot_mpi_gpu()