/**
 * test_sivf_t2i_pareto.cpp
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: T2I-1B (1M Subset) Pareto Frontier
 * Evaluates Recall@10 (Intersection) vs. QPS across various nprobe settings.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <unordered_set>

#include <faiss/IndexFlat.h> 
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include "t2i_loader.h"

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
        // T2I GT usually stores top-100, we only care about top-k
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
void run_pareto_sweep(const std::string& name, GpuIndexIVF* index, 
                      int nq, float* xq, int k, 
                      int* gt, int gt_dim) {
    
    std::vector<int> nprobes = {1, 5, 10, 20, 32, 40, 64, 80, 100, 128};

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " Pareto Benchmark: " << name << " (T2I-1M)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "nprobe\tLatency(ms)\tQPS\t\tRecall@10" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<int> I_int(nq * k);

    for (int nprobe : nprobes) {
        index->nprobe = nprobe;

        // Warmup
        index->search(100, xq, k, D.data(), I.data());
        cudaDeviceSynchronize();

        // Timing
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
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/t2i/";
    size_t nb_load = 1000000; // Load 1M
    int nlist = 1024;
    int k = 10;

    // 2. Load Data (Using T2I-specific loaders)
    size_t d, nb, nq, d_xq, ngt_dim, ngt_num;
    
    std::cout << "Loading T2I Data..." << std::endl;
    // Note: fbin_read is used for vectors, ibin_read for GT
    float* xb = fbin_read((dir + "t2i_base_1M.fbin").c_str(), &d, &nb);
    if(nb_load > nb) nb_load = nb;
    
    float* xq = fbin_read((dir + "t2i_query.fbin").c_str(), &d_xq, &nq);
    int* gt   = ibin_read((dir + "t2i_1M_gt.bin").c_str(), &ngt_dim, &ngt_num);

    std::cout << "[Info] Base: " << nb_load << ", Query: " << nq << ", Dim: " << d << std::endl;

    // 3. Resources
    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); // 512MB as requested

    // ==========================================
    // Round 1: Baseline (Standard Faiss)
    // ==========================================
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        std::cout << "\n[Baseline] Training & Adding..." << std::endl;
        index.train(50000, xb); 
        index.add(nb_load, xb);
        
        run_pareto_sweep("Baseline", &index, nq, xq, k, gt, ngt_dim);
    } 

    // Ensure cleanup
    cudaDeviceSynchronize();

    // ==========================================
    // Round 2: SIVF (Proposed)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, config);
        
        // Critical: Exact Capacity for T2I (200D) to avoid OOM
        size_t cap = nb_load; 
        
        std::cout << "\n[SIVF] Allocating exact capacity: " << cap << std::endl;
        index.initSlabManager(cap, d); 

        std::cout << "[SIVF] Training & Adding..." << std::endl;
        index.train(50000, xb);
        index.add(nb_load, xb);
        
        run_pareto_sweep("SIVF (Ours)", &index, nq, xq, k, gt, ngt_dim);
    }

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}

/** Example output:
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_t2i_pareto
Loading T2I Data...
[Loader] Header info -> N: 1000000, D: 200
[Loader] Header info -> N: 100000, D: 200
[Info] Base: 1000000, Query: 100000, Dim: 200
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[Baseline] Training & Adding...

==========================================================
 Pareto Benchmark: Baseline (T2I-1M)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       2994.05         33399           31.1484%
5       3490.24         28651           61.0131%
10      4107.44         24346           72.2472%
20      5389.86         18553           81.4864%
32      6911.82         14467           86.2951%
40      7977.69         12534           88.2616%
64      11024.22                9070            91.6871%
80      13063.47                7654            93.0067%
100     16474.17                6070            94.2459%
128     20276.65                4931            95.4052%
----------------------------------------------------------

[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF] Training & Adding...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 200D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.05 s
  Iteration 19 (0.51 s, search 0.24 s): objective=22642.1 imbalance=1.206 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

==========================================================
 Pareto Benchmark: SIVF (Ours) (T2I-1M)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       371.62          269092          31.6783%
5       1599.39         62524           61.4145%
10      2990.50         33439           72.5413%
20      5623.41         17782           81.6731%
32      8706.23         11486           86.4725%
40      10690.33                9354            88.4061%
64      16650.80                6005            91.7874%
80      20595.90                4855            93.0890%
100     25525.65                3917            94.3264%
128     32425.63                3083            95.4752%
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */