import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# ==========================================
# 1. 录入最新数据 (来自你的 Benchmark)
# ==========================================
data = [
    # NB, nlist, System, QPS
    (1000000, 1024, 'SIVF', 8884938),
    (1000000, 1024, 'Vanilla', 2089515),
    (1000000, 4096, 'SIVF', 8925076),
    (1000000, 4096, 'Vanilla', 1670939),
    (1000000, 16384, 'SIVF', 8972541),
    (1000000, 16384, 'Vanilla', 631012),
    
    (5000000, 1024, 'SIVF', 4092307),
    (5000000, 1024, 'Vanilla', 2279141),
    (5000000, 4096, 'SIVF', 9027568),
    (5000000, 4096, 'Vanilla', 1837076),
    (5000000, 16384, 'SIVF', 9048614),
    (5000000, 16384, 'Vanilla', 786301),

    (10000000, 1024, 'SIVF', 5051730),
    (10000000, 1024, 'Vanilla', 2351118),
    (10000000, 4096, 'SIVF', 8952879),
    (10000000, 4096, 'Vanilla', 1867595),
    (10000000, 16384, 'SIVF', 7759836),
    (10000000, 16384, 'Vanilla', 693278),
]

df = pd.DataFrame(data, columns=['NB', 'nlist', 'System', 'QPS'])
df['QPS_Millions'] = df['QPS'] / 1e6

# 计算 Speedup 用于热力图
pivot_sivf = df[df['System'] == 'SIVF'].set_index(['NB', 'nlist'])['QPS']
pivot_vanilla = df[df['System'] == 'Vanilla'].set_index(['NB', 'nlist'])['QPS']
speedup_df = (pivot_sivf / pivot_vanilla).reset_index()
speedup_df.rename(columns={'QPS': 'Speedup'}, inplace=True)

# ==========================================
# 2. 绘图设置 (Wide Layout for Double Column)
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif' # 论文通常用 sans-serif
plt.rcParams['pdf.fonttype'] = 42 # 确保字体可编辑

# 创建 1行3列 的画布，宽 16英寸，高 4.5英寸
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

# -------------------------------------------------------
# Subplot 1: Scalability (Throughput vs Database Size)
# 固定 nlist=4096 (最具代表性的一组)
# -------------------------------------------------------
subset_nb = df[df['nlist'] == 4096]
sns.lineplot(
    data=subset_nb, 
    x='NB', 
    y='QPS_Millions', 
    hue='System', 
    style='System', 
    markers=True, 
    dashes=False, 
    linewidth=2.5,
    markersize=9,
    palette=['#d7191c', '#2b83ba'], # 红 vs 蓝
    ax=axes[0]
)
axes[0].set_title('(a) Scalability with Data Size (nlist=4096)', fontsize=13, pad=10)
axes[0].set_xlabel('Database Size (Vectors)')
axes[0].set_ylabel('Throughput (Million vec/s)')
axes[0].set_xticks([1000000, 5000000, 10000000])
axes[0].set_xticklabels(['1M', '5M', '10M'])
axes[0].set_ylim(0, 11)
axes[0].legend(title=None, loc='center right')

# -------------------------------------------------------
# Subplot 2: Impact of Clustering (Throughput vs nlist)
# 固定 NB=10M (最大压力测试)
# -------------------------------------------------------
subset_nlist = df[df['NB'] == 10000000]
sns.barplot(
    data=subset_nlist, 
    x='nlist', 
    y='QPS_Millions', 
    hue='System', 
    palette=['#d7191c', '#2b83ba'],
    edgecolor='black',
    linewidth=0.5,
    ax=axes[1]
)
# Add value labels
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f', padding=3, fontsize=10)

axes[1].set_title('(b) Impact of Granularity (10M Vectors)', fontsize=13, pad=10)
axes[1].set_xlabel('Number of Clusters (nlist)')
axes[1].set_ylabel('Throughput (Million vec/s)')
axes[1].set_ylim(0, 12)
axes[1].legend(title=None)

# -------------------------------------------------------
# Subplot 3: Speedup Heatmap (Summary)
# -------------------------------------------------------
heatmap_data = speedup_df.pivot(index="nlist", columns="NB", values="Speedup")

# [新] 手动创建一个字符串矩阵，每个格子里写死 "xx.x x"
annot_labels = heatmap_data.applymap(lambda v: f"{v:.1f}x")

sns.heatmap(
    heatmap_data, 
    annot=annot_labels,  # [关键] 传入字符串矩阵，而不是 True
    fmt="",              # [关键] 因为是字符串了，不需要格式化，设为空
    cmap="YlGnBu", 
    linewidths=.5, 
    cbar_kws={'label': 'Speedup Factor'},
    annot_kws={"size": 12, "weight": "bold"},
    ax=axes[2]
)
axes[2].set_title('(c) Speedup Factor (ElasticIVF vs Vanilla)', fontsize=13, pad=10)
axes[2].set_xlabel('Database Size (NB)')
axes[2].set_ylabel('Number of Clusters (nlist)')
axes[2].set_xticklabels(['1M', '5M', '10M'])

# ==========================================
# Save
# ==========================================
plt.savefig('../figures/performance_add.pdf', format='pdf', dpi=300)
print("Figure saved to ../figures/performance_add.pdf")
plt.show()