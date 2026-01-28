/**
 * test_sivf_gist_pareto.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: GIST1M Pareto Frontier (Baseline vs. SIVF)
 *
 * This test evaluates the search throughput and Recall@10 on the high-dimensional
 * GIST dataset (960d) by sweeping nprobe values. It demonstrates the trade-off 
 * between the contiguous memory access of the baseline and the linked-slab 
 * architecture of SIVF.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <unordered_set>

// Essential Faiss Headers
#include <faiss/IndexFlat.h> 
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include "gist_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Compute Recall@K (Intersection)
// Returns: (Count of GT@K found in Results@K) / K
// ---------------------------------------------------------
double compute_recall_intersection(int nq, int k, const int* I, const int* gt, int gt_dim) {
    long long total_hits = 0;
    
    // Iterate over queries
    for (int i = 0; i < nq; i++) {
        std::unordered_set<int> gt_set;
        // Load Top-K Ground Truth for this query
        for (int g = 0; g < k; g++) {
            gt_set.insert(gt[i * gt_dim + g]);
        }

        // Check how many of our results are in the GT set
        for (int j = 0; j < k; j++) {
            if (gt_set.count((int)I[i * k + j])) {
                total_hits++;
            }
        }
    }
    
    return (double)total_hits / (nq * k);
}

// ---------------------------------------------------------
// Pareto Sweep Function
// ---------------------------------------------------------
void run_pareto_sweep(const std::string& name, faiss::gpu::GpuIndexIVF* index, 
                      int nq, float* xq, int k, 
                      int* gt, int gt_dim) {
    
    std::vector<int> nprobes = {1, 5, 10, 20, 32, 40, 64, 80, 100, 128};

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " Pareto Benchmark: " << name << " (GIST1M)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "nprobe\tLatency(ms)\tQPS\t\tRecall@10" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    // Result buffers
    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<int> I_int(nq * k);

    for (int nprobe : nprobes) {
        index->nprobe = nprobe;

        // Warmup
        index->search(10, xq, k, D.data(), I.data());
        cudaDeviceSynchronize();

        // Timing measurement
        auto t1 = std::chrono::high_resolution_clock::now();
        index->search(nq, xq, k, D.data(), I.data());
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time_sec = std::chrono::duration<double>(t2 - t1).count();
        double time_ms = time_sec * 1000.0;
        double qps = nq / time_sec;

        // Convert idx_t (long) to int for recall calculation
        for(size_t i=0; i<I.size(); ++i) I_int[i] = (int)I[i];

        // Compute Intersection Recall
        double recall = compute_recall_intersection(nq, k, I_int.data(), gt, gt_dim);

        std::cout << nprobe << "\t" 
                  << std::fixed << std::setprecision(2) << time_ms << "\t\t" 
                  << (size_t)qps << "\t\t" 
                  << std::setprecision(4) << recall * 100.0 << "%" << std::endl;
    }
    std::cout << "----------------------------------------------------------" << std::endl;
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

    std::cout << "[Info] Base: " << nb_load << ", Query: " << nq << ", Dim: " << d << std::endl;

    // 2. Reduce Temporary Memory Usage (2GB -> 512MB)
    // SIVF consumes significant VRAM for 960d vectors; minimize temp buffer.
    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 
    faiss::IndexFlatL2 quantizer(d);

    // ==========================================
    // Round 1: Baseline (Standard Faiss)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        std::cout << "\n[Baseline] Training & Adding..." << std::endl;
        baseline_index.train(50000, xb); 
        baseline_index.add(nb_load, xb);
        
        run_pareto_sweep("Baseline", &baseline_index, nq, xq, k, gt, ngt_dim);
    } // Baseline index is destructed here, releasing VRAM

    // Ensure cleanup
    cudaDeviceSynchronize();

    // ==========================================
    // Round 2: SIVF (Proposed)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);
        
        // Critical Optimization: Exact Capacity Allocation
        // For search benchmarks, we do not need the 1.5x buffer reserved for dynamic insertion.
        // Allocating exact capacity prevents OOM on high-dimensional data.
        size_t cap = nb_load; 
        
        std::cout << "\n[SIVF] Allocating exact capacity: " << cap << std::endl;
        sivf_index.initSlabManager(cap, d); 

        std::cout << "[SIVF] Training & Adding..." << std::endl;
        sivf_index.train(50000, xb);
        sivf_index.add(nb_load, xb);
        
        run_pareto_sweep("SIVF (Ours)", &sivf_index, nq, xq, k, gt, ngt_dim);
    }

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}

/** Example output:
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_gist_pareto
Loading GIST...
[Info] Base: 1000000, Query: 1000, Dim: 960
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[Baseline] Training & Adding...

==========================================================
 Pareto Benchmark: Baseline (GIST1M)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       58.10           17210           23.5100%
5       111.11          9000            56.6300%
10      164.08          6094            72.0600%
20      276.79          3612            85.8500%
32      403.48          2478            92.4200%
40      498.02          2007            94.4600%
64      746.65          1339            97.7700%
80      903.33          1107            98.7700%
100     1111.16         899             99.4100%
128     1384.28         722             99.7200%
----------------------------------------------------------

[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF] Training & Adding...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.22 s
  Iteration 19 (1.46 s, search 0.89 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

==========================================================
 Pareto Benchmark: SIVF (Ours) (GIST1M)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       42.89           23315           23.3600%
5       194.83          5132            56.3500%
10      378.45          2642            72.2800%
20      748.07          1336            86.0500%
32      1188.06         841             92.4600%
40      1490.26         671             94.6600%
64      2262.28         442             97.7700%
80      2825.47         353             98.6900%
100     3459.81         289             99.3100%
128     4341.17         230             99.6900%
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */