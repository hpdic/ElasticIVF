import matplotlib.pyplot as plt
import numpy as np
import os

# GPU counts: 1, 2, 3, 4, 6, 8, 10, 12
gpus = np.array([1, 2, 3, 4, 6, 8, 10, 12])

# Ingestion Total QPS (Millions)
# Updated with 12-GPU result: 45.46 M (based on nlist=2048 run)
insert_qps = np.array([2.81, 5.65, 8.72, 13.61, 24.93, 31.56, 37.63, 45.46])

# Search Total QPS (Thousands)
# Updated with 12-GPU result: 30.03 K
search_qps = np.array([2.64, 5.27, 7.92, 10.53, 15.79, 21.05, 26.35, 30.05])

# Deletion Total QPS (Millions)
# Updated with 12-GPU result: 103.17 M
delete_qps = np.array([7.68, 16.55, 23.04, 31.87, 51.54, 64.51, 77.73, 103.17])

# Visualization configuration for single-column papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 22,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'legend.fontsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# 1. Ingestion Scaling
axes[0].plot(gpus, insert_qps, 's-', color='#d62728', linewidth=4, markersize=12, label='SIVF')
axes[0].plot(gpus, insert_qps[0] * gpus, '--', color='gray', alpha=0.6, label='Ideal')
axes[0].set_title('Ingestion')
axes[0].set_ylabel('M QPS')

# 2. Search Scaling
axes[1].plot(gpus, search_qps, 'o-', color='#d62728', linewidth=4, markersize=12, label='SIVF')
axes[1].plot(gpus, search_qps[0] * gpus, '--', color='gray', alpha=0.6, label='Ideal')
axes[1].set_title('Search')
axes[1].set_ylabel('K QPS')

# 3. Deletion Scaling
axes[2].plot(gpus, delete_qps, 'd-', color='#d62728', linewidth=4, markersize=12, label='SIVF')
axes[2].plot(gpus, delete_qps[0] * gpus, '--', color='gray', alpha=0.6, label='Ideal')
axes[2].set_title('Deletion')
axes[2].set_ylabel('M QPS')

for ax in axes:
    ax.set_xlabel('Total GPUs')
    ax.set_xticks(gpus) # Ensure all GPU counts are shown on axis
    ax.grid(True, linestyle=':', alpha=0.5)

# Add legend to the last plot only to save space
axes[2].legend(loc='lower right', frameon=True)

plt.tight_layout()

# Save to the specific project directory
output_dir = os.path.expanduser('~/hpdic/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, 'scalability.pdf')
plt.savefig(output_file, bbox_inches='tight')
print(f"Scalability figure saved to {output_file}")