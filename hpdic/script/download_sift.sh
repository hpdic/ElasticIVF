#!/bin/bash

# 设置目录变量
BASE_DIR="$HOME/ElasticIVF/hpdic"
DATA_DIR="$BASE_DIR/data"
SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

# 1. 确保数据目录存在
if [ ! -d "$DATA_DIR" ]; then
    echo "[*] Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
else
    echo "[*] Data directory exists: $DATA_DIR"
fi

# 进入数据目录
cd "$DATA_DIR"

# 2. 下载数据 (检查是否已经下载)
if [ -d "sift" ]; then
    echo "[!] SIFT data folder already exists. Skipping download."
    exit 0
fi

if [ ! -f "sift.tar.gz" ]; then
    echo "[*] Downloading SIFT1M (approx 150MB)..."
    # 使用 wget 下载，如果没装 wget 可以换成 curl -O
    wget "$SIFT_URL"
else
    echo "[*] sift.tar.gz found, skipping download."
fi

# 3. 解压
echo "[*] Extracting sift.tar.gz..."
tar -zxvf sift.tar.gz

# 4. 整理 (可选：删除压缩包)
rm sift.tar.gz

echo "[*] Done. SIFT1M data is located at: $DATA_DIR/sift"
echo "    - Base vectors:  $DATA_DIR/sift/sift_base.fvecs"
echo "    - Query vectors: $DATA_DIR/sift/sift_query.fvecs"
echo "    - Ground truth:  $DATA_DIR/sift/sift_groundtruth.ivecs"