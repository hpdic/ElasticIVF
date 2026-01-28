/**
 * test_sivf_pareto_top10.cpp
 * * Modified to measure Recall@10 (Intersection of Top-10 Results with Top-10 GT).
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

// Faiss & SIVF Headers
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> 

#include "sift_loader.h"

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
    std::cout << " Pareto Benchmark: " << name << std::endl;
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
    // 1. Path Configuration
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/sift/";
    std::string base_path = dir + "sift_base.fvecs";
    std::string query_path = dir + "sift_query.fvecs";
    std::string gt_path   = dir + "sift_groundtruth.ivecs";

    // 2. Parameters
    int nlist = 1024; 
    int k = 10; // Top-10

    // 3. Load Data
    size_t d, nb, nq, ngt_dim, ngt_num;
    
    std::cout << "[Loader] Loading Base..." << std::endl;
    float* xb = fvecs_read(base_path.c_str(), &d, &nb);
    
    size_t nb_test = 1000000; // Full 1M
    size_t nt_test = 50000;   // 50k train
    if (nb_test > nb) nb_test = nb;

    std::cout << "[Loader] Loading Query..." << std::endl;
    float* xq = fvecs_read(query_path.c_str(), &d, &nq); 

    std::cout << "[Loader] Loading GroundTruth..." << std::endl;
    int* gt = ivecs_read(gt_path.c_str(), &ngt_dim, &ngt_num);

    std::cout << "[Info] Base: " << nb_test << ", Query: " << nq << ", Dim: " << d << std::endl;

    // 4. Resource Initialization
    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); 
    faiss::IndexFlatL2 quantizer(d);

    // =========================================================
    // Round 1: Baseline (Standard Faiss)
    // =========================================================
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        std::cout << "\n[Baseline] Training & Adding..." << std::endl;
        baseline_index.train(nt_test, xb);
        baseline_index.add(nb_test, xb);
        
        run_pareto_sweep("Baseline", &baseline_index, nq, xq, k, gt, ngt_dim);
    } 

    // =========================================================
    // Round 2: SIVF (Proposed)
    // =========================================================
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
 * (myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_sift_pareto
[Loader] Loading Base...
[Loader] Loading Query...
[Loader] Loading GroundTruth...
[Info] Base: 1000000, Query: 10000, Dim: 128
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[Baseline] Training & Adding...

==========================================================
 Pareto Benchmark: Baseline
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       288.33          34681           36.2320%
5       326.51          30627           74.3130%
10      378.53          26417           86.8430%
20      486.81          20541           94.8520%
32      614.24          16280           97.7100%
40      690.07          14491           98.5520%
64      948.55          10542           99.5380%
80      1088.87         9183            99.7310%
100     1389.76         7195            99.8370%
128     1699.05         5885            99.8980%
----------------------------------------------------------

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)


[SIVF] Training & Adding...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 19 (0.44 s, search 0.17 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

==========================================================
 Pareto Benchmark: SIVF (Ours)
==========================================================
nprobe  Latency(ms)     QPS             Recall@10
----------------------------------------------------------
1       30.88           323857          36.3210%
5       131.78          75885           74.1940%
10      246.19          40619           86.7690%
20      464.94          21508           94.8110%
32      721.46          13860           97.7380%
40      888.82          11250           98.5860%
64      1380.35         7244            99.5740%
80      1697.12         5892            99.7440%
100     2097.98         4766            99.8540%
128     2634.88         3795            99.9100%
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */