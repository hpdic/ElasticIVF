#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <random>
#include <omp.h>

// Faiss & SIVF Headers
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h> // 用于 GPU <-> CPU 拷贝
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> 

#include "sift_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// 辅助函数: 生成随机删除的 ID
// ---------------------------------------------------------
std::vector<faiss::idx_t> generate_delete_ids(size_t total_vectors, size_t delete_count) {
    std::vector<faiss::idx_t> ids(total_vectors);
    for(size_t i=0; i<total_vectors; ++i) ids[i] = i;
    
    // 洗牌
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    
    // 取前 N 个
    ids.resize(delete_count);
    return ids;
}

int main(int argc, char** argv) {
    // 1. 配置
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    size_t nb = 1000000; // 跑满 1M
    int nlist = 1024;    // 聚类数
    size_t n_delete = 10000; // 删除 1万条 (论文里的 Batch Size)

    // 2. 加载数据
    size_t d, file_nb;
    std::cout << "[Loader] Reading SIFT1M..." << std::endl;
    float* xb = fvecs_read(base_file, &d, &file_nb);
    if(nb > file_nb) nb = file_nb;

    // 生成对应的 ID (0, 1, 2... N-1)
    // Faiss 需要 IDSelector 或者是 ID 列表，这里我们用 IDSelectorBatch
    std::cout << "[Prepare] Generating " << n_delete << " random IDs to delete..." << std::endl;
    std::vector<faiss::idx_t> delete_ids = generate_delete_ids(nb, n_delete);
    
    // ID 选择器
    faiss::IDSelectorBatch selector(n_delete, delete_ids.data());

    // 资源
    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 512);
    faiss::IndexFlatL2 quantizer(d);

    // =========================================================
    // Round 1: Baseline (Roundtrip Deletion)
    // =========================================================
    {
        std::cout << "\n[Baseline] Setting up GPU Index..." << std::endl;
        faiss::gpu::GpuIndexIVFFlat gpu_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        // 训练 & 添加
        gpu_index.train(100000, xb); // 用前10w训练
        gpu_index.add(nb, xb);
        
        cudaDeviceSynchronize();
        std::cout << "[Baseline] Ready. Starting Deletion Benchmark (Roundtrip)..." << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();

        // --- 模拟 Faiss 不支持 GPU 删除的 workaround ---
        // 1. Copy GPU -> CPU
        faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&gpu_index);
        
        // 2. CPU Delete
        cpu_index->remove_ids(selector);
        
        // 3. Copy CPU -> GPU (重建 Index)
        // 注意：这里必须把旧的 gpu_index 覆盖或者新建一个，为了模拟完整开销，我们转回 GPU
        faiss::gpu::GpuIndexIVFFlat* new_gpu_index = 
            dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index));
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  -> Time: " << time_ms << " ms" << std::endl;
        std::cout << "  -> Throughput: " << (size_t)(n_delete / (time_ms / 1000.0)) << " vecs/sec" << std::endl;

        delete cpu_index;
        delete new_gpu_index;
    }

    // =========================================================
    // Round 2: SIVF (Native Deletion)
    // =========================================================
    {
        std::cout << "\n[SIVF] Setting up..." << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);
        
        // 初始化内存池
        sivf_index.initSlabManager(nb * 1.5, d);

        // 训练 & 添加
        sivf_index.train(100000, xb);
        sivf_index.add(nb, xb);

        cudaDeviceSynchronize();
        std::cout << "[SIVF] Ready. Starting Native Deletion..." << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();

        // --- SIVF 原生删除 ---
        // 直接在 GPU 上操作 Bitmap
        sivf_index.remove_ids(selector);

        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  -> Time: " << time_ms << " ms" << std::endl;
        std::cout << "  -> Throughput: " << (size_t)(n_delete / (time_ms / 1000.0)) << " vecs/sec" << std::endl;
    }

    delete[] xb;
    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_sift_delete
[Loader] Reading SIFT1M...
[Prepare] Generating 10000 random IDs to delete...

[Baseline] Setting up GPU Index...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Ready. Starting Deletion Benchmark (Roundtrip)...
  -> Time: 1626.24 ms
  -> Throughput: 6149 vecs/sec

[SIVF] Setting up...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 100000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 19 (0.65 s, search 0.35 s): objective=4.83016e+09 imbalance=1.135 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Ready. Starting Native Deletion...
  -> Time: 0.856534 ms
  -> Throughput: 11674959 vecs/sec
cc@rtx6000:~/ElasticIVF/build$ 
 */