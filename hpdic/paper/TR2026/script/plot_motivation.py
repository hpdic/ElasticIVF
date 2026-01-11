import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

# ================= 1. 数据准备 =================
# Python (Faiss) 实验数据 (Step 1-9, 去掉 Step 0 预热)
py_add_data = [27.65, 28.57, 27.91, 28.29, 27.86, 27.97, 28.13, 28.49, 28.55]
py_del_data = [236.63, 211.60, 212.52, 212.11, 211.61, 207.90, 210.23, 208.46, 209.89]

# C++ (Faiss) 实验数据 (Step 1-9, 去掉 Step 0 预热)
cpp_add_data = [29.86, 27.93, 27.22, 27.74, 30.99, 31.02, 27.87, 27.33, 27.09]
cpp_del_data = [210.88, 194.75, 197.75, 206.64, 201.16, 214.34, 199.68, 198.89, 196.10]

# 计算均值
py_add_mean = np.mean(py_add_data)
py_del_mean = np.mean(py_del_data)
cpp_add_mean = np.mean(cpp_add_data)
cpp_del_mean = np.mean(cpp_del_data)

# 计算 C++ 环境下的减速倍数 (作为核心论据)
slowdown_cpp = cpp_del_mean / cpp_add_mean

print(f"Python Means: Add={py_add_mean:.2f}, Del={py_del_mean:.2f}")
print(f"C++ Means:    Add={cpp_add_mean:.2f}, Del={cpp_del_mean:.2f}")

# ================= 2. 绘图配置 =================
# 设置字体为 Serif (通常论文首选)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

# 创建画布
fig, ax = plt.subplots(figsize=(6, 4.5))

# 定义 X 轴位置
labels = ['Insertion', 'Deletion']
x = np.arange(len(labels))
width = 0.35  # 柱状图宽度

# ================= 3. 绘制柱状图 =================
# Python: 使用带斜线纹理 (hatch='//') 且颜色稍淡
rects1 = ax.bar(x - width/2, [py_add_mean, py_del_mean], width, 
                label='Python (Faiss)', 
                color=['#98df8a', '#ff9896'],  # 淡绿，淡红
                edgecolor='black', alpha=0.9, hatch='//')

# C++: 使用实色，颜色更深，代表原生性能
rects2 = ax.bar(x + width/2, [cpp_add_mean, cpp_del_mean], width, 
                label='C++ (Faiss)', 
                color=['#2ca02c', '#d62728'],  # 深绿，深红
                edgecolor='black', alpha=0.9)

# ================= 4. 坐标轴与标注 =================
ax.set_ylabel('Latency (ms)  [Lower is Better]', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

# 标题：直接点出问题核心
ax.set_title('High Deletion Overhead: Python vs. C++', pad=20, fontweight='bold', fontsize=16)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_ylim(0, py_del_mean * 1.4)  # 留出顶部空间给文本标注

# 自定义图例 (为了让图例更清晰，我们手动创建 Handle)
legend_elements = [
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Python'),
    Patch(facecolor='white', edgecolor='black', label='C++'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=False)

# 数值标签函数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

autolabel(rects1)
autolabel(rects2)

# ================= 5. 核心结论标注 =================
# 在 C++ Deletion 柱子上方添加 "7.1x Slower" 的标注
center_del = x[1] + width/2 # C++ Deletion 的中心位置
height_del = cpp_del_mean
ax.text(center_del, height_del + 30, 
        f'~{slowdown_cpp:.1f}x Slower\nthan Insertion', 
        ha='center', va='bottom', 
        fontweight='bold', color='#d62728', fontsize=11)

# ================= 6. 保存图片 =================
plt.tight_layout()

# 确保输出目录存在
output_dir = os.path.expanduser("~/ElasticIVF/hpdic/paper/TR2026/figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pdf_path = os.path.join(output_dir, 'motivation_bar_chart_comparison.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300)

print(f"Figure generated and saved to: {pdf_path}")
plt.show()