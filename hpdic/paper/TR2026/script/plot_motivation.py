import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# ================= 配置 =================
# 输出路径
output_dir = os.path.expanduser('~/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
OUTPUT_PATH = os.path.join(output_dir, "motivation_bar_chart_comparison.pdf")

# 数据准备
py_add, py_del = 27.78, 207.42
cpp_add, cpp_del = 27.85, 197.94
cagra_add, cagra_del = 32.79, 3030.0

slowdown_cpp = cpp_del / cpp_add
slowdown_cagra = cagra_del / cagra_add

# ================= 绘图逻辑 =================
# 关键：设置为瘦长比例 (宽 3.5 inch, 高 3.0 inch)
plt.figure(figsize=(3.5, 3.0))

# 字体设置 (因为图变小了，字体要适配，太大会重叠，太小看不清)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,         # 基础字体
    'axes.labelsize': 10,   # 轴标签
    'axes.titlesize': 10,   # 标题
    'xtick.labelsize': 9,
    'ytick.labelsize': 8,
    'hatch.linewidth': 0.5  # 填充线变细
})

ax = plt.gca()

# X轴位置
labels = ['Insertion', 'Deletion']
x = np.arange(len(labels))
width = 0.25

# 绘制柱子
# 1. Python (IVF)
rects1 = ax.bar(x - width, [py_add, py_del], width, 
                color='#b2df8a', edgecolor='black', linewidth=0.8, hatch='////', alpha=0.9, label='IVF (Py)')

# 2. C++ (IVF)
rects2 = ax.bar(x, [cpp_add, cpp_del], width, 
                color='#33a02c', edgecolor='black', linewidth=0.8, alpha=0.9, label='IVF (C++)')

# 3. CAGRA
rects3 = ax.bar(x + width, [cagra_add, cagra_del], width, 
                color='#e31a1c', edgecolor='black', linewidth=0.8, hatch='xxxx', alpha=0.9, label='CAGRA')

# 坐标轴设置
ax.set_yscale('log')
ax.set_ylabel('Latency (ms)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(10, 20000) # 留出顶部空间给文字

# 标题 (可选，如果论文里有 Caption，图里可以不写标题以节省空间)
# ax.set_title('(a) Overhead Comparison', fontweight='bold', pad=10)

# 图例 (放在顶部，分两列，节省垂直空间)
legend_elements = [
    Patch(facecolor='#b2df8a', edgecolor='black', hatch='////', label='IVF(Py)'),
    Patch(facecolor='#33a02c', edgecolor='black', label='IVF(C++)'),
    Patch(facecolor='#e31a1c', edgecolor='black', hatch='xxxx', label='CAGRA'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
          ncol=3, fontsize=7, frameon=False, columnspacing=1.0)

# 数值标注函数
def autolabel(rects, is_cagra=False):
    for rect in rects:
        height = rect.get_height()
        # 对于 CAGRA Deletion，稍微移高一点避免和箭头重叠
        offset = 8 if (is_cagra and height > 1000) else 3
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3, is_cagra=True)

# 关键结论标注 (Slowdown)
# C++
ax.text(x[1], cpp_del * 1.5, f'{slowdown_cpp:.1f}x\nSlower', 
        ha='center', va='bottom', color='#33a02c', fontsize=8, fontweight='bold')

# CAGRA
ax.text(x[1] + width, cagra_del * 1.8, f'{slowdown_cagra:.0f}x\nSlower', 
        ha='center', va='bottom', color='#e31a1c', fontsize=8, fontweight='bold')

# 网格
ax.yaxis.grid(True, linestyle='--', alpha=0.5, which='major')

plt.tight_layout(pad=0.5) # 极度紧凑
print(f"Saving to {OUTPUT_PATH}")
plt.savefig(OUTPUT_PATH, bbox_inches='tight', pad_inches=0.02)