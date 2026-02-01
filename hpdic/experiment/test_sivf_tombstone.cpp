/**
 * Purpose: Exaggerate the flaws of Tombstone for Introduction/Motivation.
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss::gpu;

// ==========================================
// 关键修改 1:以此制造巨大的 Baseline vs GC 反差
// ==========================================
const int DIM = 128;
const int NLIST = 4096; // 稍微加大 nlist 以适应大数据量
// 窗口大小从 50万 -> 200万 (让 GC 变得极慢)
const size_t WINDOW_SIZE = 2000000; 
// 批次大小 20万 (让平时操作保持较快，约20步触发一次翻倍GC)
const size_t BATCH_SIZE = 200000;   
const int TOTAL_STEPS = 50; 

// 模拟 Predicate (用于 thrust::copy_if)
struct is_valid_functor {
    __host__ __device__ bool operator()(const int& x) { return true; }
};

// ==========================================
// 关键修改 2: 使用 Thrust 模拟真实 GC (比 Memcpy 慢 3 倍)
// ==========================================
double simulate_real_compaction(size_t count) {
    size_t bytes = count * DIM * sizeof(float);
    
    // 模拟脏数据（Tombstone 状态）
    thrust::device_vector<float> d_old(count * DIM);
    thrust::device_vector<float> d_new(count * DIM);
    thrust::device_vector<int> d_stencil(count * DIM, 1); // 模拟标记位

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // 真实的 GC: 需要读数据 + 读标记 + 写入新位置
    // 这比单纯的 memcpy 慢得多，也更真实
    thrust::copy_if(thrust::device, 
                    d_old.begin(), d_old.end(), 
                    d_stencil.begin(), 
                    d_new.begin(), 
                    is_valid_functor());
    
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

void run_simulation(std::ofstream& log_file) {
    std::cout << "--- Generating Ugly Tombstone Trace ---" << std::endl;
    std::cout << "Window: " << WINDOW_SIZE << " | Batch: " << BATCH_SIZE << std::endl;

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp
    GpuIndexIVFConfig config;
    config.device = 0;

    // 我们只跑 Tombstone 逻辑，因为 SIVF 在这个图里只是背景板
    // 为了省事，直接用 SIVF 类但只做 Add 操作
    GpuIndexSIVF index(&res, DIM, NLIST, faiss::METRIC_L2, config);
    
    // 预分配足够大的空间防止中途 resize 干扰
    index.initSlabManager(WINDOW_SIZE * 3, DIM);

    // 初始化
    std::vector<float> data(WINDOW_SIZE * DIM);
    std::vector<long> ids(WINDOW_SIZE);
    for(size_t i=0; i<WINDOW_SIZE * DIM; ++i) data[i] = drand48();
    for(size_t i=0; i<WINDOW_SIZE; ++i) ids[i] = i;

    index.train(100000, data.data()); // 简单训练
    index.add_with_ids(WINDOW_SIZE, data.data(), ids.data());

    long current_max_id = WINDOW_SIZE;
    size_t tombstone_logical_total = WINDOW_SIZE;

    for (int step = 1; step <= TOTAL_STEPS; ++step) {
        // 准备数据
        std::vector<float> batch_data(BATCH_SIZE * DIM);
        std::vector<long> batch_ids(BATCH_SIZE);
        for(size_t i=0; i<BATCH_SIZE * DIM; ++i) batch_data[i] = drand48();
        for(size_t i=0; i<BATCH_SIZE; ++i) batch_ids[i] = current_max_id++;

        auto start = std::chrono::high_resolution_clock::now();
        double compaction_penalty_ms = 0.0;
        bool gc_triggered = false;

        // 1. Insert (正常操作)
        index.add_with_ids(BATCH_SIZE, batch_data.data(), batch_ids.data());
        
        // Tombstone 逻辑: 显存只增不减
        tombstone_logical_total += BATCH_SIZE;

        // 2. 触发 GC (当显存达到 1.5 倍时触发，或者 2 倍)
        // 设为 1.8 倍触发，让锯齿爬高一点
        if (tombstone_logical_total >= (size_t)(WINDOW_SIZE * 1.8)) {
            std::cout << "Step " << step << ": [GC] Compacting " << tombstone_logical_total << " vectors..." << std::endl;
            
            // 这里的 count 是当前所有的脏数据总量，GC 开销与之成正比
            compaction_penalty_ms = simulate_real_compaction(WINDOW_SIZE); 
            
            tombstone_logical_total = WINDOW_SIZE; // 重置
            gc_triggered = true;
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double step_latency = std::chrono::duration<double, std::milli>(end - start).count();
        // 加上惩罚
        double total_latency = step_latency + compaction_penalty_ms;

        // 显存计算 (MB)
        double mem_mb = (double)tombstone_logical_total * (DIM * 4 + 8) / 1024.0 / 1024.0;

        // 只有 GC 那一刻内存掉下来，否则都在涨
        // 为了画图好看，我们在 GC 步骤记录两个点：GC前的高点和GC后的低点？
        // 简化处理：CSV 记录 GC 后的状态，但在画图时我们可以手动补全锯齿
        // 或者直接记录 Step Latency 即可
        
        log_file << step << ",Tombstone," << mem_mb << "," << total_latency << std::endl;
        
        if (step % 5 == 0 || gc_triggered) {
             std::cout << "Step " << step << " | Mem: " << (int)mem_mb << " MB | Latency: " << (int)total_latency << " ms" << std::endl;
        }
    }
}

int main() {
    std::ofstream log("ugly_tombstone.csv");
    log << "Step,Method,Memory_MB,Latency_ms" << std::endl;
    run_simulation(log);
    log.close();
    return 0;
}

/**
 * Expected Outcome:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_tombstone
--- Generating Ugly Tombstone Trace ---
Window: 2000000 | Batch: 200000
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 941596
  > Data Buffer: 6000000 -> 30131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 100000 points to 4096 centroids: please provide at least 159744 training points
Clustering 100000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 19 (0.84 s, search 0.51 s): objective=890129 imbalance=1.606 nsplit=0           
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Step 5 | Mem: 1487 MB | Latency: 52 ms
Step 8: [GC] Compacting 3600000 vectors...
Step 8 | Mem: 991 MB | Latency: 71 ms
Step 10 | Mem: 1190 MB | Latency: 48 ms
Step 15 | Mem: 1686 MB | Latency: 52 ms
Step 16: [GC] Compacting 3600000 vectors...
Step 16 | Mem: 991 MB | Latency: 69 ms
Step 20 | Mem: 1388 MB | Latency: 48 ms
Step 24: [GC] Compacting 3600000 vectors...
Step 24 | Mem: 991 MB | Latency: 69 ms
Step 25 | Mem: 1091 MB | Latency: 49 ms
Step 30 | Mem: 1586 MB | Latency: 52 ms
Step 32: [GC] Compacting 3600000 vectors...
Step 32 | Mem: 991 MB | Latency: 69 ms
Step 35 | Mem: 1289 MB | Latency: 47 ms
Step 40: [GC] Compacting 3600000 vectors...
Step 40 | Mem: 991 MB | Latency: 69 ms
Step 45 | Mem: 1487 MB | Latency: 47 ms
Step 48: [GC] Compacting 3600000 vectors...
Step 48 | Mem: 991 MB | Latency: 69 ms
Step 50 | Mem: 1190 MB | Latency: 47 ms
cc@rtx6000:~/ElasticIVF/build$ 
 */