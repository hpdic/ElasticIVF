/**
 * test_sivf_t2i_add.cpp
 * 
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 * 
 * Benchmark: T2I (Text-to-Image) Ingestion Performance
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include <faiss/IndexFlat.h>

#include "t2i_loader.h" 

using namespace faiss::gpu;

int main() {
    // 1. Configuration
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/t2i/t2i_base_1M.fbin";
    
    // Test full 1M or subset
    size_t target_nb = 1000000; 
    int nlist = 1024; // 1024 clusters is standard for 1M vectors

    size_t d, file_nb;
    std::cout << "[Loader] Reading T2I Base..." << std::endl;
    float* raw_data = fbin_read(base_file, &d, &file_nb);
    
    if(target_nb > file_nb) target_nb = file_nb;

    // 2. Resource Initialization
    StandardGpuResources res;
    // T2I vectors are usually 200d-768d. 1GB temp memory is safe.
    res.setTempMemory(1024L * 1024 * 1024); 

    size_t n_train = std::min((size_t)50000, target_nb);

    // ==========================================
    // Round 1: Baseline (Standard Faiss)
    // ==========================================
    {
        std::cout << "\n--- Baseline Training & Ingestion ---" << std::endl;
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
    // Round 2: SIVF (Proposed)
    // ==========================================
    {
        std::cout << "\n--- SIVF Training & Ingestion ---" << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf(&res, d, nlist, faiss::METRIC_L2, config);

        // Pre-allocate buffer with redundancy for dynamic resizing
        size_t cap = target_nb * 1.2; 
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
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_t2i_add
[Loader] Reading T2I Base...
[Loader] Header info -> N: 1000000, D: 200

--- Baseline Training & Ingestion ---
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Add QPS: 34596

--- SIVF Training & Ingestion ---

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200 -> 191596
  > Data Buffer: 1200000 -> 6131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 200D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 19 (0.51 s, search 0.25 s): objective=22642.1 imbalance=1.206 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Add QPS:     2908835
cc@rtx6000:~/ElasticIVF/build$ 
 */