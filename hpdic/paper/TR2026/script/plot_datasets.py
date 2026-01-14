import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 0. 配置输出路径
# ==========================================
OUTPUT_DIR = os.path.expanduser("~/ElasticIVF/hpdic/paper/TR2026/figures")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. 字体与样式配置 (大幅增大字号)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- 关键修改：字号调大 ---
plt.rcParams['font.size'] = 22          # 全局基础字号
plt.rcParams['axes.labelsize'] = 26     # X/Y轴 标签
plt.rcParams['axes.titlesize'] = 26     # 标题
plt.rcParams['xtick.labelsize'] = 22    # 刻度
plt.rcParams['ytick.labelsize'] = 22    # 刻度
plt.rcParams['legend.fontsize'] = 20    # 图例

# 颜色配置 (黑白/灰阶风格，适合打印)
# 经典学术配色 (蓝色 vs 橙色)
COLOR_BASE = '#1f77b4' # Muted Blue (Baseline)
COLOR_SIVF = '#ff7f0e' # Safety Orange (SIVF - 醒目)

# ==========================================
# 2. 数据准备
# ==========================================
datasets = ['SIFT1M (128D)', 'GIST1M (960D)']

# Ingestion (Higher is Better)
add_base = [35901, 23492]
add_sivf = [3783727, 852742]

# Deletion (Lower is Better)
del_base = [1626.0, 11843.0]
del_sivf = [0.86, 0.89]

# Search (Higher is Better)
search_base = [26702, 3640]
search_sivf = [40933, 1344]

# ==========================================
# 3. 绘图核心函数
# ==========================================

def draw_bar_chart(ylabel, data_base, data_sivf, filename_suffix, log_scale=False, mode="higher_better"):
    x = np.arange(len(datasets))
    width = 0.35  

    # 稍微加大一点画布，给大字体留空间
    fig, ax = plt.subplots(figsize=(9, 7))
    
    rects1 = ax.bar(x - width/2, data_base, width, label='Faiss Baseline', 
                    color=COLOR_BASE, alpha=0.7, edgecolor='black', hatch='//')
    rects2 = ax.bar(x + width/2, data_sivf, width, label='SIVF (Ours)', 
                    color=COLOR_SIVF, alpha=0.9, edgecolor='black')

    ax.set_ylabel(ylabel, fontweight='bold') # 轴标签加粗
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold') # 数据集名称加粗
    ax.legend(frameon=False) # 去掉图例边框，显得更干净
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 调整 Y 轴空间，给柱子上的文字留更多空余
    if log_scale:
        ax.set_yscale('log')
        # Log 模式下，上限给高一点 (50倍)，防止文字出界
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 50)
    else:
        # 线性模式下，上限给 1.4 倍
        ax.set_ylim(top=max(max(data_base), max(data_sivf)) * 1.4)

    # 自动标注数值
    def autolabel(rects, is_sivf=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # --- 数值格式化 ---
            if height >= 1000000:
                val_text = f'{height/1000000:.2f}M'
            elif height >= 1000:
                val_text = f'{height/1000:.0f}k' # 去掉小数位，更紧凑
            elif height < 10:
                val_text = f'{height:.2f}'
            else:
                val_text = f'{int(height)}'

            # 字体调大到 18
            ax.annotate(val_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=18)
            
            # --- 加速比标注 ---
            if is_sivf:
                base = data_base[i]
                curr = data_sivf[i]
                
                if mode == "lower_better": # Latency
                    speedup = base / curr
                    if speedup > 1000:
                        txt = f"{speedup/1000:.1f}k x" # 比如 13k x
                    else:
                        txt = f"{speedup:.0f}x"
                    color = 'black' # 打印友好
                else: # QPS
                    speedup = curr / base
                    if speedup < 1:
                        txt = f"{speedup:.2f}x"
                        color = 'black'
                    else:
                        txt = f"{speedup:.0f}x"
                        color = 'black'

                # 字体调大到 18 并加粗
                offset = 25 if log_scale else 30
                if mode == "lower_better": offset = 25

                ax.annotate(txt,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, offset), 
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            fontsize=20, fontweight='bold', color=color)

    autolabel(rects1)
    autolabel(rects2, is_sivf=True)

    plt.tight_layout()
    
    full_path = os.path.join(OUTPUT_DIR, f"{filename_suffix}.pdf")
    plt.savefig(full_path, dpi=300)
    print(f"[Success] Generated: {full_path}")

# ==========================================
# 4. 生成三张图
# ==========================================

# 图 1: Ingestion
draw_bar_chart(
    ylabel='Throughput (vecs/s)', # 简化标签
    data_base=add_base,
    data_sivf=add_sivf,
    filename_suffix='eval_ingestion',
    log_scale=True,
    mode="higher_better"
)

# 图 2: Deletion
draw_bar_chart(
    ylabel='Latency (ms)',
    data_base=del_base,
    data_sivf=del_sivf,
    filename_suffix='eval_deletion',
    log_scale=True,
    mode="lower_better"
)

# 图 3: Search
draw_bar_chart(
    ylabel='Query Throughput (QPS)',
    data_base=search_base,
    data_sivf=search_sivf,
    filename_suffix='eval_search',
    log_scale=False,
    mode="higher_better"
)