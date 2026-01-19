/**
 * test_sivf_deep_add.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: Deep1B Ingestion Performance (Baseline vs. SIVF)
 *
 * This test loads Deep1B vectors (96 dim), trains an index (or uses K-Means fallback),
 * and benchmarks the insertion throughput (QPS).
 */

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
#include <faiss/gpu/GpuIndexSIVF.h> // SIVF Header

// CRITICAL: Use the new loader that supports .fbin
#include "deep_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Performance Benchmark Function
// ---------------------------------------------------------
double run_benchmark(const std::string& name, faiss::Index* index, 
                     size_t n, const float* data, bool sync_gpu = true) {
    
    std::cout << "\n[Benchmark] Running " << name << "..." << std::endl;
    
    if(sync_gpu) cudaDeviceSynchronize();

    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Core Add Operation
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
    // 1. Configuration (Default: 100k vectors)
    // NOTE: Path updated to .fbin file
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/deep1b/deep1b_base_1M.fbin";
    size_t target_nb = 100000; // Adjusted to 100k for rapid testing
    int nlist = 1024;          // 1024 clusters

    if (argc > 1) target_nb = std::stoll(argv[1]);
    if (argc > 2) nlist = std::stoi(argv[2]);

    // 2. Load Data
    size_t d, file_nb;
    std::cout << "[Loader] Reading Deep1B..." << std::endl;
    
    // CRITICAL CHANGE: Use fbin_read instead of fvecs_read
    float* raw_data = fbin_read(base_file, &d, &file_nb);
    
    // Check dimension (should be 96 for Deep1B)
    std::cout << "[Info] Detected Dimension: " << d << std::endl;

    // Prepare Database (Tile/Copy if target_nb > file_nb)
    std::vector<float> database(target_nb * d);
    #pragma omp parallel for
    for (size_t i = 0; i < target_nb; ++i) {
        size_t src_idx = i % file_nb;
        std::memcpy(database.data() + i * d, raw_data + src_idx * d, d * sizeof(float));
    }
    delete[] raw_data;
    
    // Training Set (Cap at 50k for speed)
    size_t n_train = std::min((size_t)50000, target_nb); 
    std::cout << "[Info] Test Size: " << target_nb << ", Train Size: " << n_train << ", nlist: " << nlist << std::endl;

    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); // 512MB Temp Memory

    faiss::IndexFlatL2 quantizer_base(d);

    // =========================================================
    // Round 1: Baseline (Standard Faiss)
    // =========================================================
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer_base, d, nlist, faiss::METRIC_L2);
        
        std::cout << "[Baseline] Training..." << std::endl;
        baseline_index.train(n_train, database.data());
        
        run_benchmark("Faiss GPU Baseline", &baseline_index, target_nb, database.data());
    } // Index destroyed immediately to free VRAM

    // =========================================================
    // Round 2: SIVF (Ours)
    // =========================================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config; 
        config.device = 0; 

        // Constructor for SIVF
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);

        // Critical Initialization
        size_t capacity = target_nb * 1.5; // Reserve buffer
        std::cout << "[SIVF] Initializing Slab Manager (Capacity: " << capacity << ")..." << std::endl;
        
        // Pass dimension 'd' (96) to correctly calculate memory footprint
        sivf_index.initSlabManager(capacity, d); 

        // Train
        std::cout << "[SIVF] Training..." << std::endl;
        sivf_index.train(n_train, database.data());
        
        // Benchmark
        run_benchmark("SIVF (Ours)", &sivf_index, target_nb, database.data());
    }

    return 0;
}

/**
 * Example output:
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_deep_add
[Loader] Reading Deep1B...
[Loader] Reading .fbin: N=1000000, D=96
[Info] Detected Dimension: 96
[Info] Test Size: 100000, Train Size: 50000, nlist: 1024
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Training...

[Benchmark] Running Faiss GPU Baseline...
  -> Count: 100000 vectors
  -> Time:  2.74911 s
  -> QPS:   36375 vecs/sec
[SIVF] Initializing Slab Manager (Capacity: 150000)...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   96 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF] Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 96D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.21 s, search 0.14 s): objective=22677.8 imbalance=1.225 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[Benchmark] Running SIVF (Ours)...
  -> Count: 100000 vectors
  -> Time:  0.0228257 s
  -> QPS:   4381030 vecs/sec
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */