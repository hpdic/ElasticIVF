#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>

// Faiss & SIVF Headers
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> // 你的 SIVF 头文件

#include "sift_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// 辅助: 性能测试函数
// ---------------------------------------------------------
double run_benchmark(const std::string& name, faiss::Index* index, 
                     size_t n, const float* data, bool sync_gpu = true) {
    
    std::cout << "\n[Benchmark] Running " << name << "..." << std::endl;
    
    if(sync_gpu) cudaDeviceSynchronize();

    auto t_start = std::chrono::high_resolution_clock::now();
    
    // 核心 Add 操作
    index->add(n, data);
    
    if(sync_gpu) cudaDeviceSynchronize();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(t_end - t_start).count();
    
    double qps = n / time_sec;
    
    std::cout << "  -> Count: " << n << " vectors" << std::endl;
    std::cout << "  -> Time:  " << time_sec << " s" << std::endl;
    std::cout << "  -> QPS:   " << (size_t)qps << " vecs/sec" << std::endl;
    
    return qps;
}

int main(int argc, char** argv) {
    // 1. 配置参数 (默认只跑 10万)
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    size_t target_nb = 100000; // <--- 改成了 10w
    int nlist = 1024;          // <--- 10w数据配1024聚类比较合适，4096太稀疏会报错

    if (argc > 1) target_nb = std::stoll(argv[1]);
    if (argc > 2) nlist = std::stoi(argv[2]);

    // 2. 加载数据
    size_t d, file_nb;
    std::cout << "[Loader] Reading SIFT1M..." << std::endl;
    float* raw_data = fvecs_read(base_file, &d, &file_nb);
    
    // 准备数据
    std::vector<float> database(target_nb * d);
    #pragma omp parallel for
    for (size_t i = 0; i < target_nb; ++i) {
        size_t src_idx = i % file_nb;
        std::memcpy(database.data() + i * d, raw_data + src_idx * d, d * sizeof(float));
    }
    delete[] raw_data;
    
    // 训练数据 (最多取 5万 就够了，太大了训练慢)
    size_t n_train = std::min((size_t)50000, target_nb); 
    std::cout << "[Info] Test Size: " << target_nb << ", Train Size: " << n_train << ", nlist: " << nlist << std::endl;

    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); // 512MB Temp

    faiss::IndexFlatL2 quantizer_base(d);

    // =========================================================
    // Round 1: Baseline (Standard Faiss)
    // =========================================================
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer_base, d, nlist, faiss::METRIC_L2);
        
        std::cout << "[Baseline] Training..." << std::endl;
        baseline_index.train(n_train, database.data());
        
        run_benchmark("Faiss GPU Baseline", &baseline_index, target_nb, database.data());
    } // 跑完立刻销毁，释放显存

// =========================================================
    // Round 2: SIVF (Ours)
    // =========================================================
    {
        // 修正点 1: Config 类型必须是 IVFFlatConfig
        faiss::gpu::GpuIndexIVFFlatConfig config; 
        config.device = 0; 

        // 修正点 2: 构造函数去掉 &quantizer (参数列表: res, d, nlist, metric, config)
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);

        // 关键初始化
        size_t capacity = target_nb * 1.5; // 留点余量
        std::cout << "[SIVF] Initializing Slab Manager (Capacity: " << capacity << ")..." << std::endl;
        
        // 修正点 3: 补上维度 d，否则算不出内存大小
        // 如果这里还报错，请看一下你的 GpuIndexSIVF.h 里的 initSlabManager 定义
        sivf_index.initSlabManager(capacity, d); 

        // 训练
        std::cout << "[SIVF] Training..." << std::endl;
        sivf_index.train(n_train, database.data());
        
        // 测速
        run_benchmark("SIVF (Ours)", &sivf_index, target_nb, database.data());
    }

    return 0;
}

/**
 * Example output:
 cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_sift_add
[Loader] Reading SIFT1M...
[Info] Test Size: 100000, Train Size: 50000, nlist: 1024
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Training...

[Benchmark] Running Faiss GPU Baseline...
  -> Count: 100000 vectors
  -> Time:  2.78541 s
  -> QPS:   35901 vecs/sec
[SIVF] Initializing Slab Manager (Capacity: 150000)...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF] Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 19 (0.25 s, search 0.18 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[Benchmark] Running SIVF (Ours)...
  -> Count: 100000 vectors
  -> Time:  0.026429 s
  -> QPS:   3783727 vecs/sec
cc@rtx6000:~/ElasticIVF/build$ 
 */