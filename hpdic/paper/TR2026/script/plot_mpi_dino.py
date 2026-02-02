import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 样式设置 (保持原样)
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
    'figure.figsize': (17, 7.5),   
    'hatch.linewidth': 2.0        
})

UW_PURPLE = '#4b2e83' 
UW_GOLD   = '#b7a57a' 

# ==========================================
# 2. 数据准备 (更新为 12 卡数据)
# ==========================================
labels = ['Ingestion', 'Deletion']

# 更新后的 12 卡求和数据
baseline_vals = [1513201, 94352]      
sivf_vals     = [4068862, 108500942]   

# Search 数据拼接 (QPS 求和, Recall 对齐)
# Recall 保持 10 卡和 2 卡的均衡分布表现
recall_axis = [0.8075, 0.8837, 0.9344, 0.9674, 0.9871, 0.9952] 
# Vanilla QPS Sum: 10卡 + 2卡
vanilla_qps = [172602, 159235, 123862, 85038, 52434, 29616]
# SIVF QPS Sum: 10卡 + 2卡
sivf_qps    = [159331, 151684, 116687, 80505, 49982, 28914]

# ==========================================
# 3. 绘图逻辑
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2)

# -------------------------------------------------------
# Left Plot: Update Performance (12-Card)
# -------------------------------------------------------
x = np.arange(len(labels))
width = 0.45 

rects1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', 
                 color=UW_GOLD, edgecolor='black', alpha=1.0, hatch='//', linewidth=2)
rects2 = ax1.bar(x + width/2, sivf_vals, width, label='SIVF (Ours)', 
                 color=UW_PURPLE, edgecolor='black', alpha=1.0, linewidth=2)

ax1.set_yscale('log')
ax1.set_ylim(bottom=1e4, top=8e9) 

ax1.set_ylabel('Throughput (QPS)', fontweight='bold')
ax1.set_title('(a) Update Performance (12-Card)', fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold', fontsize=26)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=22)
ax1.grid(axis='y', linestyle='--', alpha=0.5, which='major', linewidth=1.5)

def label_bars_log(rects, is_sivf=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height > 1e6: val_str = f'{height/1e6:.1f}M'
        elif height > 1e3: val_str = f'{height/1e3:.0f}K'
        else: val_str = f'{height:.0f}'
        
        ax1.annotate(val_str,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=22, fontweight='bold', color='#333')
        
        if is_sivf:
            speedup = height / baseline_vals[i]
            # Deletion 的 Speedup 超过 1000 倍，这里做个处理
            label_text = f"{int(speedup)}x" if speedup >= 100 else f"{speedup:.1f}x"
            ax1.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 45), textcoords="offset points", 
                        ha='center', va='bottom', fontsize=26, fontweight='bold', color=UW_PURPLE)

label_bars_log(rects1)
label_bars_log(rects2, is_sivf=True)

# -------------------------------------------------------
# Right Plot: Search Performance (Pareto Frontier)
# -------------------------------------------------------
ax2.plot(recall_axis, vanilla_qps, 'o--', color=UW_GOLD, label='Baseline', 
         linewidth=4.5, markersize=16, markerfacecolor='white', markeredgewidth=3.5)
ax2.plot(recall_axis, sivf_qps, 's-', color=UW_PURPLE, label='SIVF (Ours)', 
         linewidth=4.5, markersize=16, alpha=0.9)

ax2.set_xlabel('Recall@10', fontweight='bold')
ax2.set_ylabel('Total QPS', fontweight='bold') 
ax2.set_title('(b) Search Performance (12-Card)', fontweight='bold', pad=20)
ax2.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax2.legend(fontsize=24)

# Annotation Box - 强调在大规模下的性能维持
ax2.text(0.88, 120000, "Maintain\nHigh Search\nEfficiency", 
         fontsize=24, color='#333', ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=UW_GOLD, lw=3, alpha=0.9))

# ==========================================
# 4. 保存
# ==========================================
plt.tight_layout()
output_dir = os.path.expanduser('~/hpdic/ElasticIVF/hpdic/paper/TR2026/figures/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.subplots_adjust(wspace=0.25) 
save_path = os.path.join(output_dir, 'mpi_dino.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"12-GPU Plot saved to {save_path}")