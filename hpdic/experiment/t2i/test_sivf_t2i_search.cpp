/**
 * test_sivf_t2i_search.cpp
 * 
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: T2I Search Performance (Latency & Recall)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <faiss/IndexFlat.h> 
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include "t2i_loader.h"

using namespace faiss::gpu;

double calc_recall(int nq, int k, const int* I, const int* gt, int gt_dim) {
    int ok = 0;
    for (int i = 0; i < nq; i++) {
        // Assume GT structure: [nq * gt_dim]. First element is NN.
        // We check if any of top-k retrieved (I) matches the top-1 GT.
        int true_id = gt[i * gt_dim]; 
        for (int j = 0; j < k; j++) {
            if (I[i * k + j] == true_id) { ok++; break; }
        }
    }
    return (double)ok / nq;
}

void bench(const char* name, GpuIndexIVF* index, int nq, float* xq, int k, int* gt, int gt_dim) {
    index->nprobe = 20; 
    
    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);
    
    // Warmup
    index->search(10, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    index->search(nq, xq, k, D.data(), I.data());
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    double time = std::chrono::duration<double>(t2 - t1).count();
    
    std::vector<int> I_int(nq * k);
    for(size_t i=0; i<I.size(); ++i) I_int[i] = (int)I[i];
    
    double recall = calc_recall(nq, k, I_int.data(), gt, gt_dim);
    std::cout << "[" << name << "] QPS: " << (size_t)(nq/time) 
              << " | Recall@10: " << recall * 100.0 << "%" << std::endl;
}

int main() {
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/t2i/";
    size_t nb_load = 1000000;
    int nlist = 1024;
    int k = 10;

    size_t d, nb, nq, d_xq, ngt_dim, ngt_num;
    
    std::cout << "Loading T2I Data..." << std::endl;
    float* xb = fbin_read((dir + "t2i_base_1M.fbin").c_str(), &d, &nb);
    if(nb_load > nb) nb_load = nb;
    
    // Query file
    float* xq = fbin_read((dir + "t2i_query.fbin").c_str(), &d_xq, &nq);
    
    // GT file
    int* gt   = ibin_read((dir + "t2i_1M_gt.bin").c_str(), &ngt_dim, &ngt_num);

    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 

    // Baseline
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        index.train(50000, xb); 
        index.add(nb_load, xb);
        bench("Baseline", &index, nq, xq, k, gt, ngt_dim);
    }

    // SIVF
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, config);
        
        index.initSlabManager(nb_load, d); // Exact capacity

        index.train(50000, xb);
        index.add(nb_load, xb);
        bench("SIVF", &index, nq, xq, k, gt, ngt_dim);
    }

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_t2i_search 
Loading T2I Data...
[Loader] Header info -> N: 1000000, D: 200
[Loader] Header info -> N: 100000, D: 200
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] QPS: 18635 | Recall@10: 84.152%

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 200D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.05 s
  Iteration 19 (0.32 s, search 0.23 s): objective=22642.1 imbalance=1.206 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] QPS: 17796 | Recall@10: 84.355%
cc@rtx6000:~/ElasticIVF/build$ 
 */