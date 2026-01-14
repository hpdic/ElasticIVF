import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. Data Preparation
# ==========================================
# Baseline (Faiss) ~ 202.2 ms
raw_baseline = [202.2, 201.8, 202.5] 
# SIVF (Ours) ~ 0.685 ms
raw_sivf = [0.685, 0.692, 0.678]

data_groups = [raw_baseline, raw_sivf]
methods = ['Faiss IVF\n(Baseline)', 'SIVF\n(Ours)']

lat_means = [np.mean(g) for g in data_groups]
lat_stds = [np.std(g) for g in data_groups]

# Throughput calculation
tp_groups = [[10000 / (x / 1000) for x in g] for g in data_groups]
tp_means = [np.mean(g) for g in tp_groups]
tp_stds = [np.std(g) for g in tp_groups]

speedup_val = lat_means[0] / lat_means[1]

# ==========================================
# 2. Plotting Setup
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
colors = ['#4c72b0', '#c44e52']

save_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# [Adjustment 1] figsize=(8, 5) -> Wider, less "thin/tall"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
plt.subplots_adjust(wspace=0.15, bottom=0.15, top=0.92) # Reduce gap between plots

# ==========================================
# Subplot 1: Latency
# ==========================================
bars1 = ax1.bar(methods, lat_means, yerr=lat_stds, capsize=4, 
               color=colors, width=0.6, edgecolor='black', linewidth=1,
               error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

ax1.set_yscale('log')
ax1.set_ylabel('Deletion Latency (ms) [Log Scale]', fontweight='bold')
ax1.set_ylim(bottom=0.1) 

# Value Label (SIVF)
sivf_bar = bars1[1]
h_sivf = sivf_bar.get_height()
ax1.text(sivf_bar.get_x() + sivf_bar.get_width()/2., h_sivf * 1.15, 
         f'{h_sivf:.2f} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Speedup Arrow
ax1.annotate(f'{speedup_val:.1f}x Speedup', 
             xy=(1.0, h_sivf * 2.5), 
             xytext=(0.9, lat_means[0] * 0.1), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=11, fontweight='bold', ha='center', color='black')

# Label (a)
ax1.text(0.5, -0.15, '(a) Deletion Latency', transform=ax1.transAxes, 
         ha='center', va='top', fontweight='bold', fontsize=13)

# ==========================================
# Subplot 2: Throughput (Right Y-Axis)
# ==========================================
bars2 = ax2.bar(methods, tp_means, yerr=tp_stds, capsize=4, 
                color=colors, width=0.6, edgecolor='black', linewidth=1,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

ax2.set_yscale('log')
ax2.set_ylabel('Throughput (vecs/sec) [Log Scale]', fontweight='bold')

# [Adjustment 2] Move Y-axis to the right
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")

# Increase Top Limit
ax2.set_ylim(bottom=min(tp_means) * 0.1, top=max(tp_means) * 20)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.15,
            f'{height:.1e}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# [Adjustment 3] Updated Label
ax2.text(0.5, -0.15, '(b) Deletion Throughput', transform=ax2.transAxes, 
         ha='center', va='top', fontweight='bold', fontsize=13)

# Save
save_path = os.path.join(save_dir, 'performance_delete.pdf')
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Generated '{save_path}'")