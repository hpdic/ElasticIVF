"""
motivation_bar_chart_comparison.py

Author: Dongfang Zhao
Email:  dzhao@uw.edu

Visualization script for the Motivation Section.
This chart compares the latency of Insertion vs. Deletion operations in Faiss
across both Python and C++ environments. It empirically demonstrates the 
high overhead of deletion (due to CPU-GPU synchronization and data movement),
which is approximately 7x slower than insertion.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# ================= 1. Data Preparation =================
# Experimental Data: Python (Faiss)
# Latency values from Steps 1-9 (Step 0 excluded as warmup)
py_add_data = [27.65, 28.57, 27.91, 28.29, 27.86, 27.97, 28.13, 28.49, 28.55]
py_del_data = [236.63, 211.60, 212.52, 212.11, 211.61, 207.90, 210.23, 208.46, 209.89]

# Experimental Data: C++ (Faiss)
# Latency values from Steps 1-9 (Step 0 excluded as warmup)
cpp_add_data = [29.86, 27.93, 27.22, 27.74, 30.99, 31.02, 27.87, 27.33, 27.09]
cpp_del_data = [210.88, 194.75, 197.75, 206.64, 201.16, 214.34, 199.68, 198.89, 196.10]

# Calculate Mean Latency
py_add_mean = np.mean(py_add_data)
py_del_mean = np.mean(py_del_data)
cpp_add_mean = np.mean(cpp_add_data)
cpp_del_mean = np.mean(cpp_del_data)

# Calculate Slowdown Factor in C++ Environment (Core Motivation Argument)
# Measures how much slower deletion is compared to insertion
slowdown_cpp = cpp_del_mean / cpp_add_mean

print(f"Python Means: Add={py_add_mean:.2f}, Del={py_del_mean:.2f}")
print(f"C++ Means:    Add={cpp_add_mean:.2f}, Del={cpp_del_mean:.2f}")

# ================= 2. Plotting Configuration =================
# Font Configuration: Serif (Standard for academic publications)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# Create Figure
fig, ax = plt.subplots(figsize=(6, 4.5))

# Define X-axis positions
labels = ['Insertion', 'Deletion']
x = np.arange(len(labels))
width = 0.35  # Bar width

# ================= 3. Render Bar Chart =================
# Series 1: Python (Faiss)
# Visual style: Hatched texture ('//') and lighter pastel colors
rects1 = ax.bar(x - width/2, [py_add_mean, py_del_mean], width, 
                label='Python (Faiss)', 
                color=['#98df8a', '#ff9896'],  # Pastel Green, Pastel Red
                edgecolor='black', alpha=0.9, hatch='//')

# Series 2: C++ (Faiss)
# Visual style: Solid, darker colors to represent native/optimized performance
rects2 = ax.bar(x + width/2, [cpp_add_mean, cpp_del_mean], width, 
                label='C++ (Faiss)', 
                color=['#2ca02c', '#d62728'],  # Deep Green, Deep Red
                edgecolor='black', alpha=0.9)

# ================= 4. Axes and Annotations =================
ax.set_ylabel('Latency (ms)  [Lower is Better]', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

# Title: Explicitly state the core problem
ax.set_title('High Deletion Overhead: Python vs. C++', pad=20, fontweight='bold', fontsize=16)

# Grid and Limits
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_ylim(0, py_del_mean * 1.4)  # Reserve top space for text annotations

# Custom Legend
# Manually create handles for clearer distinction between environments
legend_elements = [
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Python'),
    Patch(facecolor='white', edgecolor='black', label='C++'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False)

# Helper function to label bar values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

autolabel(rects1)
autolabel(rects2)

# ================= 5. Key Insight Annotation =================
# Highlight the ~7.1x slowdown on the C++ Deletion bar
center_del = x[1] + width/2 # Center position of C++ Deletion bar
height_del = cpp_del_mean
ax.text(center_del, height_del + 30, 
        f'~{slowdown_cpp:.1f}x Slower\nthan Insertion', 
        ha='center', va='bottom', 
        fontweight='bold', color='#d62728', fontsize=11)

# ================= 6. Save Figure =================
plt.tight_layout()

# Ensure output directory exists
output_dir = os.path.expanduser("~/ElasticIVF/hpdic/paper/TR2026/figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pdf_path = os.path.join(output_dir, 'motivation_bar_chart_comparison.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300)

print(f"Figure generated and saved to: {pdf_path}")