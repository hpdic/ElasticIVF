/**
 * test_sivf_sift_search.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: SIFT1M Search Performance (Baseline vs. SIVF)
 *
 * This test loads the SIFT1M dataset, builds indices for both Vanilla Faiss
 * and SIVF, and measures Search Throughput (QPS) and Recall@1.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <cmath>

// Faiss & SIVF Headers
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> 

#include "sift_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Compute Recall
// R@K: Checks if the Ground Truth 1-NN is present in the Top-K results.
// ---------------------------------------------------------
double compute_recall_at_1(int nq, int k, const int* I, const int* gt, int gt_dim) {
    int n_ok = 0;
    for (int i = 0; i < nq; i++) {
        int true_nn = gt[i * gt_dim]; // The 0-th element of GT is the 1-NN
        for (int j = 0; j < k; j++) {
            if (I[i * k + j] == true_nn) {
                n_ok++;
                break;
            }
        }
    }
    return (double)n_ok / nq;
}

// ---------------------------------------------------------
// Benchmark Execution Function
// ---------------------------------------------------------
void benchmark_search(const std::string& name, faiss::gpu::GpuIndexIVF* index, 
                      int nq, const float* xq, int k, int nprobe,
                      const int* gt, int gt_dim) {
    
    std::cout << "\n[Benchmark] " << name << " (nprobe=" << nprobe << ", k=" << k << ")" << std::endl;
    
    // Set nprobe (direct member modification is allowed in Faiss GPU indices)
    index->nprobe = nprobe;
    
    // Result buffers
    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    // Warmup run
    index->search(100, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();

    // Timing measurement
    auto t1 = std::chrono::high_resolution_clock::now();
    index->search(nq, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(t2 - t1).count();
    double qps = nq / time;

    // Convert idx_t (int64) -> int for Recall calculation (SIFT GT is int32)
    std::vector<int> I_int(nq * k);
    for(size_t i=0; i<I.size(); ++i) I_int[i] = (int)I[i];

    // Compute Recall@1
    double recall = compute_recall_at_1(nq, k, I_int.data(), gt, gt_dim);

    std::cout << "  -> Time:   " << time * 1000.0 << " ms" << std::endl;
    std::cout << "  -> QPS:    " << (size_t)qps << " queries/sec" << std::endl;
    std::cout << "  -> Recall: " << recall * 100.0 << " %" << std::endl;
}

int main(int argc, char** argv) {
    // 1. Path Configuration
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/sift/";
    std::string base_path = dir + "sift_base.fvecs";
    std::string query_path = dir + "sift_query.fvecs";
    std::string gt_path   = dir + "sift_groundtruth.ivecs";

    // 2. Parameters
    int nlist = 1024; // 1024 clusters suitable for 1M or subset vectors
    int k = 10;       // Retrieve Top-10
    int nprobe = 10;  // Search 10 clusters (buckets)

    // 3. Load Data
    size_t d, nb, nq, ngt_dim, ngt_num;
    
    std::cout << "[Loader] Loading Base..." << std::endl;
    float* xb = fvecs_read(base_path.c_str(), &d, &nb);
    
    // Limit dataset size for this specific test
    size_t nb_test = 1000000;
    size_t nt_test = 50000; // Training subset size
    if (nb_test > nb) nb_test = nb;

    std::cout << "[Loader] Loading Query..." << std::endl;
    float* xq = fvecs_read(query_path.c_str(), &d, &nq); 

    std::cout << "[Loader] Loading GroundTruth..." << std::endl;
    int* gt = ivecs_read(gt_path.c_str(), &ngt_dim, &ngt_num);

    std::cout << "[Info] Base: " << nb_test << ", Query: " << nq << ", Dim: " << d << std::endl;

    // 4. Resource Initialization
    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); // 512MB Temp Memory
    faiss::IndexFlatL2 quantizer(d);

    // =========================================================
    // Round 1: Baseline (Standard Faiss)
    // =========================================================
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        std::cout << "[Baseline] Training & Adding..." << std::endl;
        baseline_index.train(nt_test, xb);
        baseline_index.add(nb_test, xb);
        
        // Execute Search Benchmark
        benchmark_search("Baseline", &baseline_index, nq, xq, k, nprobe, gt, ngt_dim);
    } // Baseline index is destructed here to release VRAM

    // =========================================================
    // Round 2: SIVF (Proposed)
    // =========================================================
    {
        // Configuration
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        
        // Instantiate SIVF (without quantizer passed in constructor)
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);

        // Initialize Memory Pool (Slab Manager)
        // Pass dimension 'd' to ensure correct memory stride calculations
        sivf_index.initSlabManager(nb_test * 1.5, d);

        std::cout << "[SIVF] Training & Adding..." << std::endl;
        sivf_index.train(nt_test, xb);
        sivf_index.add(nb_test, xb);

        // Execute Search Benchmark
        benchmark_search("SIVF (Ours)", &sivf_index, nq, xq, k, nprobe, gt, ngt_dim);
    }

    // Cleanup Host Memory
    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_sift_search
[Loader] Loading Base...
[Loader] Loading Query...
[Loader] Loading GroundTruth...
[Info] Base: 1000000, Query: 10000, Dim: 128
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Training & Adding...

[Benchmark] Baseline (nprobe=10, k=10)
  -> Time:   374.5 ms
  -> QPS:    26702 queries/sec
  -> Recall: 90.89 %

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

[SIVF] Training & Adding...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 19 (0.23 s, search 0.16 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[Benchmark] SIVF (Ours) (nprobe=10, k=10)
  -> Time:   244.3 ms
  -> QPS:    40933 queries/sec
  -> Recall: 90.63 %
cc@rtx6000:~/ElasticIVF/build$ 
 */