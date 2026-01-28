"""
motivation_bar_chart_comparison.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for the Motivation Section.
This chart compares the latency of Insertion vs. Deletion operations across:
1. Python Faiss (IVF)
2. C++ Faiss (IVF)
3. GPU CAGRA (Graph-based)

It empirically demonstrates that while IVF suffers from ~7x deletion slowdown,
Graph-based indices (CAGRA) suffer from catastrophic ~92x slowdown due to 
reconstruction requirements, validating the need for SIVF.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# ================= 1. Data Preparation =================

# --- Dataset 1: Python Faiss (IVF) ---
# Source: python3 ~/ElasticIVF/hpdic/script/benchmark_baseline.py
py_add_data = [27.34, 28.70, 27.56, 27.52, 27.31, 28.37, 28.20, 27.98, 27.78]
py_del_data = [235.01, 206.95, 204.71, 208.51, 205.22, 209.72, 207.42, 206.04, 205.68]
py_add_mean = np.mean(py_add_data)
py_del_mean = np.mean(py_del_data)

# --- Dataset 2: C++ Faiss (IVF) ---
# Source: ./benchmark_baseline.bin
cpp_add_data = [27.81, 27.68, 28.04, 28.06, 27.81, 27.83, 27.49, 27.85, 27.79]
cpp_del_data = [197.57, 197.67, 197.37, 197.42, 197.78, 198.75, 197.95, 197.94, 198.92]
cpp_add_mean = np.mean(cpp_add_data)
cpp_del_mean = np.mean(cpp_del_data)

# --- Dataset 3: GPU CAGRA (Graph) ---
# Source: Table 1 (Landscape Full)
# Insertion: 305 K vec/s -> Convert to ms for 10k batch
# Calculation: (10,000 vectors / 305,000 vec/s) * 1000 ms/s
cagra_add_mean = (10000 / 305000) * 1000  # approx 32.79 ms
cagra_del_mean = 3030.0 # From Table

# Calculate Slowdown Factors
slowdown_cpp = cpp_del_mean / cpp_add_mean
slowdown_cagra = cagra_del_mean / cagra_add_mean

print(f"GPU IVF (Python): Add={py_add_mean:.2f}, Del={py_del_mean:.2f}")
print(f"GPU IVF (C++):    Add={cpp_add_mean:.2f}, Del={cpp_del_mean:.2f}")
print(f"GPU CAGRA:        Add={cagra_add_mean:.2f}, Del={cagra_del_mean:.2f}")

# ================= 2. Plotting Configuration =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# Create Figure
fig, ax = plt.subplots(figsize=(8, 5))

# Define X-axis positions
labels = ['Insertion', 'Deletion']
x = np.arange(len(labels))
width = 0.25  # Bar width adjusted for 3 bars

# ================= 3. Render Bar Chart =================
# We have 3 bars per group.
# Positions: Center-Left, Center, Center-Right

# 1. Python Faiss (IVF) - Light Colors, Hatched
rects1 = ax.bar(x - width, [py_add_mean, py_del_mean], width, 
                label='GPU IVF (Python)', 
                color=['#98df8a', '#ff9896'],  # Pastel Green, Pastel Red
                edgecolor='black', alpha=0.9, hatch='//')

# 2. C++ Faiss (IVF) - Solid Colors
rects2 = ax.bar(x, [cpp_add_mean, cpp_del_mean], width, 
                label='GPU IVF (C++)', 
                color=['#2ca02c', '#d62728'],  # Medium Green, Medium Red
                edgecolor='black', alpha=0.9)

# 3. GPU CAGRA - Dark Colors, Cross Hatched
rects3 = ax.bar(x + width, [cagra_add_mean, cagra_del_mean], width, 
                label='GPU CAGRA', 
                color=['#006400', '#8b0000'],  # Dark Green, Dark Red
                edgecolor='black', alpha=0.9, hatch='xx')

# ================= 4. Axes and Annotations =================
# Use Log Scale because CAGRA deletion (3000ms) dwarfs Insertion (28ms)
ax.set_yscale('log')
ax.set_ylabel('Latency (ms)  [Log Scale]', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

# Title
ax.set_title('Addition and Deletion Overhead on GPUs', pad=20, fontweight='bold', fontsize=16)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0, which='major')
ax.yaxis.grid(True, linestyle=':', alpha=0.3, zorder=0, which='minor')

# Y-Axis Limits (Adjust for annotations space)
# Log scale requires careful limit setting to show the top annotations
ax.set_ylim(10, 10000) 

# Custom Legend
legend_elements = [
    Patch(facecolor='white', edgecolor='black', hatch='//', label='GPU IVF (Python)'),
    Patch(facecolor='white', edgecolor='black', label='GPU IVF (C++)'),
    Patch(facecolor='white', edgecolor='black', hatch='xx', label='GPU CAGRA'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True)

# Helper function to label bar values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # Vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# ================= 5. Key Insight Annotation =================
# Annotation for C++ IVF
ax.text(x[1], cpp_del_mean * 1.4, 
        f'~{slowdown_cpp:.1f}x\nSlower', 
        ha='center', va='bottom', 
        fontweight='bold', color='#d62728', fontsize=10)

# Annotation for CAGRA (The dramatic slowdown)
ax.text(x[1] + width, cagra_del_mean * 1.4, 
        f'~{slowdown_cagra:.0f}x\nSlower!', 
        ha='center', va='bottom', 
        fontweight='bold', color='#8b0000', fontsize=10)

# ================= 6. Save Figure =================
plt.tight_layout()

# Ensure output directory exists
output_dir = os.path.expanduser("~/ElasticIVF/hpdic/paper/TR2026/figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# [CRITICAL] Use the exact filename requested by user
pdf_path = os.path.join(output_dir, 'motivation_bar_chart_comparison.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300)

print(f"Figure generated and saved to: {pdf_path}")