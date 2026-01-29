"""
plot_mpi_split.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization for HPDC Evaluation (SIVF Scalability Only).
Generates two compact figures for Single-Column Side-by-Side layout:
1. mpi_io.pdf: SIVF Insertion & Deletion Throughput
2. mpi_search.pdf: SIVF Search Throughput
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# ============================================================
# 1. Experimental Data (SIVF Only)
# ============================================================
gpus = [1, 2, 4]

# I/O Data (Million QPS)
sivf_insert = [4.425, 11.027, 17.749]
sivf_delete = [13.908, 34.050, 64.006]

# Search Data (Thousands QPS - k QPS)
sivf_search = [5.851, 11.636, 23.344]

# ============================================================
# 2. Plotting Configuration
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
# Large fonts for small figure sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 9
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

save_dir = os.path.expanduser('~/hpdic/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Colors
c_ins = '#1f77b4' # Blue
c_del = '#d62728' # Red
c_sch = '#ff7f0e' # Orange

def plot_io():
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    
    # Plot SIVF Lines
    ax.plot(gpus, sivf_delete, marker='^', label='Deletion', color=c_del, linestyle='-')
    ax.plot(gpus, sivf_insert, marker='o', label='Insertion', color=c_ins, linestyle='-')
    
    # Styling
    ax.set_ylabel('Throughput (M QPS)', fontweight='bold')
    ax.set_xlabel('# GPUs', fontweight='bold')
    
    # Ticks
    ax.set_xticks(gpus)
    ax.set_xticklabels([str(g) for g in gpus])
    
    # Y-axis range padding for visual comfort
    ax.set_ylim(0, 70)
    
    # Grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    ax.legend(frameon=False, loc='upper left')

    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mpi_io.pdf'), bbox_inches='tight')
    print("Saved mpi_io.pdf")

def plot_search():
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    
    # Plot SIVF Line
    ax.plot(gpus, sivf_search, marker='s', label='Search', color=c_sch, linestyle='-')
    
    # Styling
    ax.set_ylabel('Throughput (K QPS)', fontweight='bold')
    ax.set_xlabel('# GPUs', fontweight='bold')
    
    # Ticks
    ax.set_xticks(gpus)
    ax.set_xticklabels([str(g) for g in gpus])
    
    # Start Y from 0 to show true scaling
    ax.set_ylim(0, 30)
    
    # Grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    ax.legend(frameon=False, loc='upper left')

    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mpi_search.pdf'), bbox_inches='tight')
    print("Saved mpi_search.pdf")

if __name__ == "__main__":
    plot_io()
    plot_search()