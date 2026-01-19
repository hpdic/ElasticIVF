/**
 * test_sivf_deep_search.cpp
 * Dataset: Deep1B (96 dim) - 1M Subset with Recomputed GT
 * 
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 * 
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <cmath>

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> 

#include "deep_loader.h"

using namespace faiss::gpu;

double compute_recall_at_1(int nq, int k, const int* I, const int* gt, int gt_dim) {
    int n_ok = 0;
    for (int i = 0; i < nq; i++) {
        // In your newly generated .bin GT, the 0-th element is the nearest neighbor
        int true_nn = gt[i * gt_dim]; 
        for (int j = 0; j < k; j++) {
            if (I[i * k + j] == true_nn) {
                n_ok++;
                break;
            }
        }
    }
    return (double)n_ok / nq;
}

void benchmark_search(const std::string& name, faiss::gpu::GpuIndexIVF* index, 
                      int nq, const float* xq, int k, int nprobe,
                      const int* gt, int gt_dim) {
    std::cout << "\n[Benchmark] " << name << " (nprobe=" << nprobe << ", k=" << k << ")" << std::endl;
    index->nprobe = nprobe;
    
    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    // Warmup
    index->search(100, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    index->search(nq, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(t2 - t1).count();
    double qps = nq / time;

    std::vector<int> I_int(nq * k);
    for(size_t i=0; i<I.size(); ++i) I_int[i] = (int)I[i];

    double recall = compute_recall_at_1(nq, k, I_int.data(), gt, gt_dim);

    std::cout << "  -> Time:   " << time * 1000.0 << " ms" << std::endl;
    std::cout << "  -> QPS:    " << (size_t)qps << " queries/sec" << std::endl;
    std::cout << "  -> Recall: " << recall * 100.0 << " %" << std::endl;
}

int main(int argc, char** argv) {
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/deep1b/";
    
    // [Path Updates]
    std::string base_path = dir + "deep1b_base_1M.fbin";
    std::string query_path = dir + "deep1b_query.fbin";
    std::string gt_path   = dir + "deep1b_groundtruth.bin";

    int nlist = 1024; 
    int k = 10;       
    int nprobe = 10;  

    size_t d, nb, nq, ngt_dim, ngt_num;
    
    std::cout << "[Loader] Loading Base..." << std::endl;
    float* xb = fbin_read(base_path.c_str(), &d, &nb);
    
    // nb should now be 1000000 exactly
    size_t nb_test = nb; 
    size_t nt_test = 50000; 

    std::cout << "[Loader] Loading Query..." << std::endl;
    float* xq = fbin_read(query_path.c_str(), &d, &nq); 

    std::cout << "[Loader] Loading GroundTruth..." << std::endl;
    // 使用 ibin_read 读取 .bin (int32)
    int* gt = ibin_read(gt_path.c_str(), &ngt_dim, &ngt_num);

    std::cout << "[Info] Base: " << nb_test << ", Query: " << nq << ", Dim: " << d << std::endl;

    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 
    faiss::IndexFlatL2 quantizer(d);

    // --- Baseline ---
    {
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        std::cout << "[Baseline] Training & Adding..." << std::endl;
        baseline_index.train(nt_test, xb);
        baseline_index.add(nb_test, xb);
        benchmark_search("Baseline", &baseline_index, nq, xq, k, nprobe, gt, ngt_dim);
    } 

    // --- SIVF ---
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);

        sivf_index.initSlabManager(nb_test * 1.5, d);

        std::cout << "[SIVF] Training & Adding..." << std::endl;
        sivf_index.train(nt_test, xb);
        sivf_index.add(nb_test, xb);

        benchmark_search("SIVF (Ours)", &sivf_index, nq, xq, k, nprobe, gt, ngt_dim);
    }

    delete[] xb;
    delete[] xq;
    delete[] gt;

    return 0;
}

/** 
 * Example Output:
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_deep_search
[Loader] Loading Base...
[Loader] Reading .fbin: N=1000000, D=96
[Loader] Loading Query...
[Loader] Reading .fbin: N=10000, D=96
[Loader] Loading GroundTruth...
[Loader] Reading GT .bin: Queries=10000, K=100
[Info] Base: 1000000, Query: 10000, Dim: 96
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Training & Adding...

[Benchmark] Baseline (nprobe=10, k=10)
  -> Time:   345.865 ms
  -> QPS:    28913 queries/sec
  -> Recall: 92.03 %

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   96 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

[SIVF] Training & Adding...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 96D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.20 s, search 0.14 s): objective=22677.8 imbalance=1.225 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[Benchmark] SIVF (Ours) (nprobe=10, k=10)
  -> Time:   167.258 ms
  -> QPS:    59787 queries/sec
  -> Recall: 91.94 %
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */