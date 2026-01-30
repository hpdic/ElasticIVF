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
#include <string>

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
void bench(const char* name, GpuIndexIVF* index, int nprobe, int nq, float* xq, int k, int* gt, int gt_dim) {
    // Set nprobe dynamically based on input argument
    index->nprobe = nprobe; 
    
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
    std::cout << "[" << name << "] nprobe: " << nprobe << " | QPS: " << (size_t)(nq/time) 
              << " | Recall@10: " << recall * 100.0 << "%" << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 5) {
        std::cout << "Usage: ./test_sivf_gist_search <nlist> <nprobe> <nt_train> <temp_mem_mb>" << std::endl;
        return 1;
    }

    int nlist = std::stoi(argv[1]);
    int nprobe = std::stoi(argv[2]);
    size_t nt_train = std::stoul(argv[3]);
    int temp_mem_mb = std::stoi(argv[4]);

    // 1. Configuration and Data Loading
    std::string dir = "/home/cc/ElasticIVF/hpdic/data/gist/";
    size_t d, nb, nq, ngt_dim, ngt_num;
    
    std::cout << "Loading GIST..." << std::endl;
    float* xb = fvecs_read((dir + "gist_base.fvecs").c_str(), &d, &nb);
    float* xq = fvecs_read((dir + "gist_query.fvecs").c_str(), &d, &nq);
    int* gt   = ivecs_read((dir + "gist_groundtruth.ivecs").c_str(), &ngt_dim, &ngt_num);

    // 2. Resource Initialization
    // Adjust temporary memory to avoid OOM on high-dimensional data
    StandardGpuResources res;
    res.setTempMemory(temp_mem_mb * 1024 * 1024); 

    // ==========================================
    // Round 1: Baseline (Standard Faiss)
    // ==========================================
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        // Train on the specified number of vectors
        index.train(nt_train, xb); 
        index.add(nb, xb);
        bench("Baseline", &index, nprobe, nq, xq, 10, gt, ngt_dim);
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
        // Allocating exact capacity prevents OOM on high-dimensional data by 
        // avoiding the default 1.5x buffer reserved for dynamic insertion.
        size_t cap = nb; 
        
        std::cout << "[SIVF] Allocating exact capacity: " << cap << std::endl;
        index.initSlabManager(cap, d); 

        index.train(nt_train, xb);
        index.add(nb, xb);
        bench("SIVF", &index, nprobe, nq, xq, 10, gt, ngt_dim);
    }

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}