import time
import sys
import os
import numpy as np
import faiss

# ================= 配置区域 =================
DATA_DIR = os.path.expanduser("~/ElasticIVF/hpdic/data/sift")
BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
LEARN_FILE = os.path.join(DATA_DIR, "sift_learn.fvecs")

DIM = 128
NLIST = 1024          # 聚类中心数量
WINDOW_SIZE = 100000  # 窗口大小 10w
BATCH_SIZE = 10000    # 每次滑动 1w
DEVICE_ID = 0         # RTX 6000 ID

# ================= 工具函数 =================
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def get_memory_usage():
    # 简单的显存占用估算 (仅作参考)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(DEVICE_ID)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2
    except:
        return 0

# ================= 主流程 =================
def main():
    print(f"Loading data from {DATA_DIR}...")
    if not os.path.exists(BASE_FILE):
        print(f"Error: {BASE_FILE} not found.")
        return

    xb = fvecs_read(BASE_FILE)
    xt = fvecs_read(LEARN_FILE)
    print(f"Data loaded. Base shape: {xb.shape}")

    # 1. 初始化 GPU
    print("Initializing Faiss GPU resources...")
    res = faiss.StandardGpuResources()
    
    # 2. 构建并训练索引 (使用 IVFFlat)
    print("Training IndexIVFFlat on CPU...")
    quantizer = faiss.IndexFlatL2(DIM)
    cpu_index = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_L2)
    cpu_index.train(xt)

    # 3. 搬运到 GPU
    print("Moving index to GPU...")
    # co = faiss.GpuClonerOptions()
    # co.useFloat16 = True # 如果需要省显存可以开启
    gpu_index = faiss.index_cpu_to_gpu(res, DEVICE_ID, cpu_index)

    # 4. 预填充窗口
    print(f"Pre filling window with {WINDOW_SIZE} vectors...")
    gpu_index.add(xb[:WINDOW_SIZE])
    
    # 记录当前在索引里的 IDs (Faiss 默认 ID 是连续的 0..N)
    # 我们用一个简单的 FIFO 队列逻辑来模拟 ID
    current_min_id = 0
    current_max_id = WINDOW_SIZE
    
    print("\n>>> Starting Streaming Benchmark <<<")
    print(f"Window: {WINDOW_SIZE}, Batch: {BATCH_SIZE}")
    print(f"{'Step':<5} | {'Add(ms)':<10} | {'Remove(ms)':<12} | {'Method':<15} | {'Total(ms)':<10}")
    print("="*65)

    # 运行 10 轮即可看出趋势
    for step in range(10):
        # 准备数据
        start_idx = WINDOW_SIZE + step * BATCH_SIZE
        if start_idx + BATCH_SIZE > xb.shape[0]:
            break
            
        new_data = xb[start_idx : start_idx + BATCH_SIZE]
        
        # 这一批要删除的 IDs (模拟最老的 ID)
        ids_to_remove = np.arange(current_min_id, current_min_id + BATCH_SIZE, dtype=np.int64)

        # A. 测试 Add
        t0 = time.time()
        gpu_index.add(new_data)
        t1 = time.time()
        add_time = (t1 - t0) * 1000

        # B. 测试 Remove
        t2 = time.time()
        method_name = "Direct"
        try:
            # 尝试直接在 GPU 上删除
            gpu_index.remove_ids(ids_to_remove)
        except Exception as e:
            # 如果报错 (NotImplemented)，则模拟 "CPU Roundtrip"
            # 这是目前 Faiss 用户被迫使用的慢速方案
            method_name = "CPU_Roundtrip"
            
            # 1. 把索引拷回 CPU
            tmp_cpu_index = faiss.index_gpu_to_cpu(gpu_index)
            # 2. 在 CPU 上删除
            tmp_cpu_index.remove_ids(ids_to_remove)
            # 3. 重新拷回 GPU (极慢)
            gpu_index = faiss.index_cpu_to_gpu(res, DEVICE_ID, tmp_cpu_index)
            
        t3 = time.time()
        remove_time = (t3 - t2) * 1000
        
        total_time = add_time + remove_time
        
        print(f"{step:<5} | {add_time:<10.2f} | {remove_time:<12.2f} | {method_name:<15} | {total_time:<10.2f}")

        # 更新 ID 计数
        current_min_id += BATCH_SIZE
        current_max_id += BATCH_SIZE

if __name__ == "__main__":
    main()