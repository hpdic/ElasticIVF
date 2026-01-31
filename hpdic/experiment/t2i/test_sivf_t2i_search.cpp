/**
 * test_sivf_t2i_search.cpp
 * * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: T2I Search Performance (Latency & Recall)
 * Usage: ./test_sivf_t2i_search [nlist=1024] [nprobe=20] [nb_load=1000000] [temp_mem_mb=512]
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>

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

// [MODIFIED] Added nprobe as an argument
void bench(const char* name, GpuIndexIVF* index, int nq, float* xq, int k, int* gt, int gt_dim, int nprobe) {
    index->nprobe = nprobe; 
    
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
    
    // [MODIFIED] Output format to match GIST script parsing
    // Format: [SIVF] nprobe: 20 | QPS: 12345 | Recall@10: 85.123%
    std::cout << "[" << name << "] nprobe: " << nprobe 
              << " | QPS: " << (size_t)(nq/time) 
              << " | Recall@10: " << recall * 100.0 << "%" << std::endl;
}

int main(int argc, char* argv[]) {
    // [MODIFIED] Parse command line arguments
    int nlist = (argc > 1) ? atoi(argv[1]) : 1024;
    int nprobe = (argc > 2) ? atoi(argv[2]) : 20;
    size_t nb_load = (argc > 3) ? atol(argv[3]) : 1000000;
    long temp_mem_mb = (argc > 4) ? atol(argv[4]) : 512;

    std::string dir = "/home/cc/ElasticIVF/hpdic/data/t2i/";
    int k = 10;
    size_t d, nb, nq, d_xq, ngt_dim, ngt_num;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Configuration: nlist=" << nlist 
              << ", nprobe=" << nprobe 
              << ", nb_load=" << nb_load 
              << ", temp_mem=" << temp_mem_mb << "MB" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::cout << "Loading T2I Data..." << std::endl;
    float* xb = fbin_read((dir + "t2i_base_1M.fbin").c_str(), &d, &nb);
    if(nb_load > nb) nb_load = nb;
    
    // Query file
    float* xq = fbin_read((dir + "t2i_query.fbin").c_str(), &d_xq, &nq);
    
    // GT file
    int* gt   = ibin_read((dir + "t2i_1M_gt.bin").c_str(), &ngt_dim, &ngt_num);

    StandardGpuResources res;
    res.setTempMemory(temp_mem_mb * 1024 * 1024); 

    // Baseline
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        // Train with a subset (e.g., 50k or 10% of data)
        int nt_train = 50000; 
        if (nt_train > nb_load) nt_train = nb_load;

        index.train(nt_train, xb); 
        index.add(nb_load, xb);
        bench("Baseline", &index, nq, xq, k, gt, ngt_dim, nprobe);
    }

    // SIVF
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, config);
        
        index.initSlabManager(nb_load, d); // Exact capacity

        int nt_train = 50000;
        if (nt_train > nb_load) nt_train = nb_load;

        index.train(nt_train, xb);
        index.add(nb_load, xb);
        bench("SIVF", &index, nq, xq, k, gt, ngt_dim, nprobe);
    }

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}