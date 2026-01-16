/**
 * test_sivf_gist_search.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: GIST1M Search Performance (Baseline vs. SIVF)
 *
 * This test evaluates the search throughput and recall on the high-dimensional
 * GIST dataset (960d). It demonstrates the trade-off between the contiguous
 * memory access of the baseline and the linked-slab architecture of SIVF.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

// Essential Faiss Headers
#include <faiss/IndexFlat.h> 
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include "gist_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Calculate Recall@K
// ---------------------------------------------------------
double calc_recall(int nq, int k, const int* I, const int* gt, int gt_dim) {
    int ok = 0;
    for (int i = 0; i < nq; i++) {
        // GIST Ground Truth stores the nearest neighbor at index 0
        int true_id = gt[i * gt_dim];
        for (int j = 0; j < k; j++) {
            if (I[i * k + j] == true_id) { ok++; break; }
        }
    }
    return (double)ok / nq;
}

// ---------------------------------------------------------
// Benchmark Execution Function
// ---------------------------------------------------------
void bench(const char* name, GpuIndexIVF* index, int nq, float* xq, int k, int* gt, int gt_dim) {
    // Set nprobe slightly higher for high-dimensional GIST data
    index->nprobe = 20; 
    
    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);
    
    // Warmup Search
    index->search(10, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();

    // Benchmark Search
    auto t1 = std::chrono::high_resolution_clock::now();
    index->search(nq, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    double time = std::chrono::duration<double>(t2 - t1).count();
    
    // Convert idx_t (long) to int for recall calculation
    std::vector<int> I_int(nq * k);
    for(size_t i=0; i<I.size(); ++i) I_int[i] = (int)I[i];
    
    double recall = calc_recall(nq, k, I_int.data(), gt, gt_dim);
    std::cout << "[" << name << "] QPS: " << (size_t)(nq/time) << " | Recall@10: " << recall * 100.0 << "%" << std::endl;
}

int main() {
    // 1. Configuration
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/gist/";
    size_t nb_load = 1000000; // Load full 1M vectors
    int nlist = 1024;
    int k = 10;

    size_t d, nb, nq, ngt_dim, ngt_num;
    std::cout << "Loading GIST..." << std::endl;
    float* xb = fvecs_read((dir + "gist_base.fvecs").c_str(), &d, &nb);
    if(nb_load > nb) nb_load = nb;
    float* xq = fvecs_read((dir + "gist_query.fvecs").c_str(), &d, &nq);
    int* gt   = ivecs_read((dir + "gist_groundtruth.ivecs").c_str(), &ngt_dim, &ngt_num);

    // 2. Reduce Temporary Memory Usage (2GB -> 512MB)
    // SIVF consumes significant VRAM for 960d vectors; minimize temp buffer.
    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 

    // ==========================================
    // Round 1: Baseline (Standard Faiss)
    // ==========================================
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        // Train on a subset (50k is sufficient)
        index.train(50000, xb); 
        index.add(nb_load, xb);
        bench("Baseline", &index, nq, xq, k, gt, ngt_dim);
    } // Baseline index is destructed here, releasing VRAM

    // Ensure cleanup
    cudaDeviceSynchronize();

    // ==========================================
    // Round 2: SIVF (Proposed)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, config);
        
        // Critical Optimization: Exact Capacity Allocation
        // For search benchmarks, we do not need the 1.5x buffer reserved for dynamic insertion.
        // Allocating exact capacity prevents OOM on high-dimensional data.
        size_t cap = nb_load; 
        
        std::cout << "[SIVF] Allocating exact capacity: " << cap << std::endl;
        index.initSlabManager(cap, d); 

        index.train(50000, xb);
        index.add(nb_load, xb);
        bench("SIVF", &index, nq, xq, k, gt, ngt_dim);
    }

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_gist_search
Loading GIST...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] QPS: 3640 | Recall@10: 89%
[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.23 s
  Iteration 19 (1.45 s, search 1.02 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] QPS: 1344 | Recall@10: 89.5%
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_gist_search
Loading GIST...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] QPS: 3599 | Recall@10: 89%
[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.26 s
  Iteration 19 (1.47 s, search 1.01 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] QPS: 1342 | Recall@10: 89.5%
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_gist_search
Loading GIST...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] QPS: 3615 | Recall@10: 89%
[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.23 s
  Iteration 19 (1.46 s, search 1.03 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] QPS: 1349 | Recall@10: 89.5%
cc@rtx6000:~/ElasticIVF/build$ 
 */