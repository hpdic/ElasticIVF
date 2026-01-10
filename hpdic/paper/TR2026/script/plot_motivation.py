import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 数据准备 =================
add_mean = 28.16   
remove_mean = 212.66 
slowdown = remove_mean / add_mean

# ================= 绘图配置 =================
# 使用通用 serif 字体避免报错
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(5, 4))

# 【核心修改】极简标签，去掉误导性的硬件标注
labels = ['Insertion', 'Deletion']
means = [add_mean, remove_mean]
x_pos = [0, 1]
bar_width = 0.5
colors = ['#2ca02c', '#d62728'] # 绿快红慢

# 绘制
bars = ax.bar(x_pos, means, color=colors, edgecolor='black', width=bar_width, alpha=0.8)

# 坐标轴
ax.set_ylabel('Latency (ms)  [Lower is Better]')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=16) # 加大标签字体
# 【核心修改】标题直接点题
ax.set_title('The High Cost of Deletion', pad=15, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, remove_mean * 1.3)

# 数值标注
ax.text(0, add_mean + 5, f'{add_mean:.1f} ms', 
        ha='center', va='bottom', color='black', fontsize=12)
ax.text(1, remove_mean + 5, f'{remove_mean:.1f} ms', 
        ha='center', va='bottom', color='black', fontsize=12)

# 核心对比标注
ax.text(1, remove_mean + 30, f'{slowdown:.1f}x Slower!', 
        ha='center', va='bottom', fontweight='bold', color='#d62728', fontsize=16)

# ================= 保存 =================
plt.tight_layout()

output_dir = os.path.expanduser("~/ElasticIVF/hpdic/paper/TR2026/figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pdf_path = os.path.join(output_dir, 'motivation_bar_chart.pdf')
plt.savefig(pdf_path, format='pdf', dpi=300)

print(f"Fixed figure saved to: {pdf_path}")