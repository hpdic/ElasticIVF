#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include <faiss/IndexFlat.h>
#include "gist_loader.h"

using namespace faiss::gpu;

int main() {
    // 路径
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
    size_t target_nb = 100000; // 先测 10万 看看速度 (全量跑太久可以改)
    int nlist = 1024;

    size_t d, file_nb;
    std::cout << "[Loader] Reading GIST1M Base..." << std::endl;
    float* raw_data = fvecs_read(base_file, &d, &file_nb);
    std::cout << "  Dim: " << d << " (High Dim!), Total in file: " << file_nb << std::endl;

    // 截取测试数据
    if(target_nb > file_nb) target_nb = file_nb;
    
    // 资源
    StandardGpuResources res;
    res.setTempMemory(1024L * 1024 * 1024 * 2); // 给 2GB Temp，GIST 很大

    // Train 数据 (5万够了)
    size_t n_train = std::min((size_t)50000, target_nb);

    // ==========================================
    // Round 1: Baseline Add
    // ==========================================
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat baseline(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        baseline.train(n_train, raw_data);
        
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        baseline.add(target_nb, raw_data);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double time = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "[Baseline] Add QPS: " << (size_t)(target_nb / time) << std::endl;
    }

    // ==========================================
    // Round 2: SIVF Add
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf(&res, d, nlist, faiss::METRIC_L2, config);

        // 关键：初始化内存池 (GIST 每个向量大，Slab要大一点)
        size_t cap = target_nb * 1.5;
        sivf.initSlabManager(cap, d);

        sivf.train(n_train, raw_data);

        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        sivf.add(target_nb, raw_data);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "[SIVF] Add QPS:     " << (size_t)(target_nb / time) << std::endl;
    }

    delete[] raw_data;
    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_gist_add
[Loader] Reading GIST1M Base...
  Dim: 960 (High Dim!), Total in file: 1000000
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Add QPS: 23492

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.14 s
  Iteration 19 (1.55 s, search 1.02 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Add QPS:     852742
cc@rtx6000:~/ElasticIVF/build$ 
 */