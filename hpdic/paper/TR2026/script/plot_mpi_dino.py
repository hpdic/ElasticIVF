import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os

# ==========================================
# 1. 样式设置
# ==========================================
plt.rcParams.update({
    'font.size': 24,              
    'font.family': 'serif',
    'font.serif': ['Times New Roman'], 
    'axes.labelsize': 28,         
    'axes.titlesize': 30,         
    'xtick.labelsize': 24,        
    'ytick.labelsize': 24,
    'legend.fontsize': 22,        
    'figure.figsize': (17, 7.5),   # 保持瘦长比例
    'hatch.linewidth': 2.0        
})

# --- UW Color Palette ---
UW_PURPLE = '#4b2e83' 
UW_GOLD   = '#b7a57a' 

# ==========================================
# 2. 数据准备
# ==========================================
labels = ['Ingestion', 'Deletion']
baseline_vals = [1496738, 77889]      
sivf_vals     = [3402566, 89462837]   

recall_axis = [0.8075, 0.8837, 0.9344, 0.9674, 0.9871, 0.9952] 
vanilla_qps = [87119, 81393, 63285, 43509, 26898, 15225]
sivf_qps    = [83908, 77410, 59630, 41315, 25771, 14714]

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2)

# -------------------------------------------------------
# Left Plot: Update Performance
# -------------------------------------------------------
x = np.arange(len(labels))
width = 0.45 

rects1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', 
                 color=UW_GOLD, edgecolor='black', alpha=1.0, hatch='//', linewidth=2)
rects2 = ax1.bar(x + width/2, sivf_vals, width, label='SIVF (Ours)', 
                 color=UW_PURPLE, edgecolor='black', alpha=1.0, linewidth=2)

ax1.set_yscale('log')
ax1.set_ylim(top=8e9) 

ax1.set_ylabel('Throughput (QPS)', fontweight='bold')
ax1.set_title('(a) Update Performance', fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold', fontsize=26)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=22)
ax1.grid(axis='y', linestyle='--', alpha=0.5, which='major', linewidth=1.5)

# 标注函数
def label_bars_log(rects, is_sivf=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        
        # 数值格式化
        if height > 1e6: val_str = f'{height/1e6:.1f}M'
        elif height > 1e3: val_str = f'{height/1e3:.0f}K'
        else: val_str = f'{height:.0f}'
        
        # 基础数值标签
        ax1.annotate(val_str,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=22, fontweight='bold', color='#333')
        
        # Speedup 标签 (去掉 "Speedup" 字样，只留倍数)
        if is_sivf:
            speedup = height / baseline_vals[i]
            label_text = f"{int(speedup)}x" if speedup > 100 else f"{speedup:.1f}x"
            ax1.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 45), textcoords="offset points", 
                        ha='center', va='bottom', fontsize=26, fontweight='bold', color=UW_PURPLE)

label_bars_log(rects1)
label_bars_log(rects2, is_sivf=True)

# -------------------------------------------------------
# Right Plot: Search Performance
# -------------------------------------------------------
ax2.plot(recall_axis, vanilla_qps, 'o--', color=UW_GOLD, label='Baseline', 
         linewidth=4.5, markersize=16, markerfacecolor='white', markeredgewidth=3.5)
ax2.plot(recall_axis, sivf_qps, 's-', color=UW_PURPLE, label='SIVF (Ours)', 
         linewidth=4.5, markersize=16, alpha=0.9)

ax2.set_xlabel('Recall@10', fontweight='bold')
# 【修改】去掉 Y 轴标签
ax2.set_ylabel('') 
ax2.set_title('(b) Search Performance', fontweight='bold', pad=20)
ax2.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax2.legend(fontsize=24)

# Annotation Box 
ax2.text(0.855, 25000, "Comparable\nPerformance\n(>95% QPS)", 
         fontsize=24, color='#333', ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=UW_GOLD, lw=3, alpha=0.9))

# ==========================================
# 4. 保存
# ==========================================
plt.tight_layout()
output_dir = os.path.expanduser('~/hpdic/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.subplots_adjust(wspace=0.15) 
save_path = os.path.join(output_dir, 'mpi_dino.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")