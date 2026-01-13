import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# ==========================================
# 1. 录入真实测量数据 (2026-01-12 Search Benchmark)
# ==========================================
data = [
    # NB, nlist, System, QPS
    (100000, 1024, 'SIVF', 280611),
    (100000, 1024, 'Vanilla', 529435),
    (100000, 4096, 'SIVF', 417376),
    (100000, 4096, 'Vanilla', 568740),
    (100000, 16384, 'SIVF', 310710),
    (100000, 16384, 'Vanilla', 326045),

    (200000, 1024, 'SIVF', 166108),
    (200000, 1024, 'Vanilla', 352114),
    (200000, 4096, 'SIVF', 272699),
    (200000, 4096, 'Vanilla', 446427),
    (200000, 16384, 'SIVF', 216029),
    (200000, 16384, 'Vanilla', 309570),

    (500000, 1024, 'SIVF', 71866),
    (500000, 1024, 'Vanilla', 177202),
    (500000, 4096, 'SIVF', 111601),
    (500000, 4096, 'Vanilla', 166162),
    (500000, 16384, 'SIVF', 104455),
    (500000, 16384, 'Vanilla', 214485),
]

df = pd.DataFrame(data, columns=['NB', 'nlist', 'System', 'QPS'])
df['QPS_Thousands'] = df['QPS'] / 1000

# 计算效率比 (SIVF / Vanilla)
pivot_sivf = df[df['System'] == 'SIVF'].set_index(['NB', 'nlist'])['QPS']
pivot_vanilla = df[df['System'] == 'Vanilla'].set_index(['NB', 'nlist'])['QPS']
relative_df = (pivot_sivf / pivot_vanilla).reset_index()
relative_df.rename(columns={'QPS': 'Relative'}, inplace=True)

# ==========================================
# 2. 绘图设置
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

# -------------------------------------------------------
# Subplot 1: Scalability (nlist=4096)
# -------------------------------------------------------
subset_nb = df[df['nlist'] == 4096]
sns.lineplot(
    data=subset_nb, 
    x='NB', 
    y='QPS_Thousands', 
    hue='System', 
    style='System', 
    markers=True, 
    dashes=False, 
    linewidth=2.5,
    markersize=9,
    palette=['#d7191c', '#2b83ba'],
    ax=axes[0]
)
axes[0].set_title('(a) Search Scalability (nlist=4096)', fontsize=13, pad=10)
axes[0].set_xlabel('Database Size (Vectors)')
axes[0].set_ylabel('Search Throughput (10^3 QPS)')
axes[0].set_xticks([100000, 200000, 500000])
axes[0].set_xticklabels(['100k', '200k', '500k'])
axes[0].set_ylim(0, 700)
axes[0].legend(title=None)

# -------------------------------------------------------
# Subplot 2: Impact of Clustering (500k Vectors)
# -------------------------------------------------------
subset_nlist = df[df['NB'] == 500000]
sns.barplot(
    data=subset_nlist, 
    x='nlist', 
    y='QPS_Thousands', 
    hue='System', 
    palette=['#d7191c', '#2b83ba'],
    edgecolor='black',
    linewidth=0.5,
    ax=axes[1]
)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.0f', padding=3, fontsize=10)

axes[1].set_title('(b) Search Granularity (500k Vectors)', fontsize=13, pad=10)
axes[1].set_xlabel('Number of Clusters (nlist)')
axes[1].set_ylabel('Search Throughput (10^3 QPS)')
axes[1].set_ylim(0, 300)
axes[1].legend(title=None)

# -------------------------------------------------------
# Subplot 3: Efficiency Ratio Heatmap
# -------------------------------------------------------
heatmap_data = relative_df.pivot(index="nlist", columns="NB", values="Relative")
annot_labels = heatmap_data.applymap(lambda v: f"{v:.2f}x")

sns.heatmap(
    heatmap_data, 
    annot=annot_labels, 
    fmt="",
    cmap="RdYlGn", 
    linewidths=.5, 
    cbar_kws={'label': 'Efficiency (SIVF/Vanilla)'},
    annot_kws={"size": 12, "weight": "bold"},
    ax=axes[2]
)
axes[2].set_title('(c) Efficiency Ratio (Search QPS)', fontsize=13, pad=10)
axes[2].set_xlabel('Database Size (NB)')
axes[2].set_ylabel('Number of Clusters (nlist)')
axes[2].set_xticklabels(['100k', '200k', '500k'])

# ==========================================
# Save
# ==========================================
plt.savefig('../figures/performance_search.pdf', format='pdf', dpi=300)
print("Figure saved to ../figures/performance_search.pdf")
plt.show()