/**
 * test_sivf_deep_pareto.cpp
 * Dataset: Deep1B (96 dim) - 1M Subset
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 * Description:
 * Generates the Pareto Frontier (Recall@10 vs. QPS) for Deep1B-1M.
 * Compares standard GPU IVF (Baseline) vs. SIVF.
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

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> 

#include "deep_loader.h"

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
                      int nq, const float* xq, int k, 
                      const int* gt, int gt_dim) {
    
    std::vector<int> nprobes = {1, 5, 10, 20, 32, 40, 64, 80, 100, 128};

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " Pareto Benchmark: " << name << " (Deep1B-1M)" << std::endl;
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
        index->search(100, xq, k, D.data(), I.data());
        cudaDeviceSynchronize();

        // Timing measurement
        auto t1 = std::chrono::high_resolution_clock::now();
        index->search(nq, xq, k, D.data(), I.data());
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time_sec = std::chrono::duration<double>(t2 - t1).count();
        double time_ms = time_sec * 1000.0;
        double qps = nq / time_sec;

        // Convert idx_t -> int for calculation
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

int main(int argc, char** argv) {
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/deep1b/";
    
    // File Paths
    std::string base_path = dir + "deep1b_base_1M.fbin";
    std::string query_path = dir + "deep1b_query.fbin";
    std::string gt_path   = dir + "deep1b_groundtruth.bin";

    int nlist = 1024; 
    int k = 10;       

    size_t d, nb, nq, ngt_dim, ngt_num;
    
    std::cout << "[Loader] Loading Base..." << std::endl;
    float* xb = fbin_read(base_path.c_str(), &d, &nb);
    
    size_t nb_test = nb;    // 1M
    size_t nt_test = 50000; // Train on 50k

    std::cout << "[Loader] Loading Query..." << std::endl;
    float* xq = fbin_read(query_path.c_str(), &d, &nq); 

    std::cout << "[Loader] Loading GroundTruth..." << std::endl;
    int* gt = ibin_read(gt_path.c_str(), &ngt_dim, &ngt_num);

    std::cout << "[Info] Base: " << nb_test << ", Query: " << nq << ", Dim: " << d << std::endl;

    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 
    faiss::IndexFlatL2 quantizer(d);

    // ==========================================
    // Round 1: Baseline (Standard Faiss)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        std::cout << "\n[Baseline] Training & Adding..." << std::endl;
        baseline_index.train(nt_test, xb);
        baseline_index.add(nb_test, xb);
        
        run_pareto_sweep("Baseline", &baseline_index, nq, xq, k, gt, ngt_dim);
    } 

    // ==========================================
    // Round 2: SIVF (Proposed)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);

        sivf_index.initSlabManager(nb_test * 1.5, d);

        std::cout << "\n[SIVF] Training & Adding..." << std::endl;
        sivf_index.train(nt_test, xb);
        sivf_index.add(nb_test, xb);

        run_pareto_sweep("SIVF (Ours)", &sivf_index, nq, xq, k, gt, ngt_dim);
    }

    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}

/** Example output:
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_deep_pareto
[Loader] Loading Base...
[Loader] Reading .fbin: N=1000000, D=96
[Loader] Loading Query...
[Loader] Reading .fbin: N=10000, D=96
[Loader] Loading GroundTruth...
[Loader] Reading GT .bin: Queries=10000, K=100
[Info] Base: 1000000, Query: 10000, Dim: 96
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[Baseline] Training & Adding...

==========================================================
 Pareto Benchmark: Baseline (Deep1B-1M)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       280.79          35613           43.6330%
5       309.87          32271           79.3080%
10      343.72          29093           89.2380%
20      421.34          23733           95.2110%
32      505.13          19796           97.4430%
40      554.54          18032           98.1750%
64      738.52          13540           99.1580%
80      854.96          11696           99.4520%
100     1065.20         9387            99.6630%
128     1317.86         7588            99.8020%
----------------------------------------------------------

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   96 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)


[SIVF] Training & Adding...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 96D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.38 s, search 0.13 s): objective=22677.8 imbalance=1.225 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

==========================================================
 Pareto Benchmark: SIVF (Ours) (Deep1B-1M)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       22.29           448640          43.6470%
5       90.82           110107          79.4150%
10      166.86          59929           89.2540%
20      309.42          32318           95.2370%
32      472.47          21165           97.4860%
40      578.97          17272           98.1920%
64      889.99          11236           99.1790%
80      1091.50         9161            99.4460%
100     1341.93         7451            99.6500%
128     1685.15         5934            99.8040%
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */