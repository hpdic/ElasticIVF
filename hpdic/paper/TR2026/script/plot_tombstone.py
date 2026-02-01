import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置 =================
output_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
OUTPUT_PATH = os.path.join(output_dir, "motivation_tombstone.pdf")

# 数据参数
BASE_LATENCY = 48.0
GC_SLOPE = 6.4
SIVF_LATENCY = 60.0
index_sizes = np.linspace(0, 100, 200)

y_sivf = [SIVF_LATENCY for _ in index_sizes]
y_tombstone = [BASE_LATENCY + (size * GC_SLOPE) for size in index_sizes]

# ================= 绘图逻辑 =================
# 关键：与图1保持完全一致的尺寸 (3.5 x 3.0)
plt.figure(figsize=(3.5, 3.0))

sns.set(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8
})

ax = plt.gca()

# 绘制曲线
ax.plot(index_sizes, y_sivf, color='#33a02c', linewidth=2.5, label='SIVF')
ax.fill_between(index_sizes, 0, y_sivf, color='#33a02c', alpha=0.1)
ax.plot(index_sizes, y_tombstone, color='#e31a1c', linewidth=2.5, linestyle='--', label='Tombstone')

# 关键点 A: 50M
x1, y1 = 50, 368
ax.vlines(x=x1, ymin=0, ymax=y1, colors='gray', linestyles=':', linewidth=1)
ax.scatter([x1], [y1], color='#e31a1c', s=30, zorder=5)
# 紧凑标注
ax.annotate(f'{y1}ms\n(Laggy)', 
             xy=(x1, y1), xytext=(x1 + 15, y1 - 80),
             arrowprops=dict(facecolor='#e31a1c', arrowstyle='->', lw=1),
             color='#e31a1c', fontsize=8, fontweight='bold', ha='center')

# 关键点 B: 100M
x2, y2 = 100, 688
ax.vlines(x=x2, ymin=0, ymax=y2, colors='black', linestyles=':', linewidth=1)
ax.scatter([x2], [y2], color='black', s=30, zorder=5)
# 紧凑标注
ax.annotate(f'{y2}ms\n(Freeze)', 
             xy=(x2, y2), xytext=(x2 - 25, y2 + 5),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
             color='black', fontsize=8, fontweight='bold', ha='center')

# 轴标签
ax.set_xlabel("Index Scale (Millions)", fontweight='bold')
ax.set_ylabel("Latency (ms)", fontweight='bold')
# ax.set_title('(b) Scalability Projection', fontweight='bold', pad=10)

# 范围控制
ax.set_xlim(0, 105)
ax.set_ylim(0, 800)

# 图例 (紧凑放置)
ax.legend(loc='upper left', frameon=True, fontsize=8, framealpha=0.9, borderpad=0.3)

# 网格
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=0.5)
print(f"Saving to {OUTPUT_PATH}")
plt.savefig(OUTPUT_PATH, bbox_inches='tight', pad_inches=0.02)