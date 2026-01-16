/**
 * test_sivf_gist_delete.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: GIST1M Deletion Performance (Roundtrip vs. Native)
 *
 * This test evaluates the deletion latency on high dimensional data (960d).
 * It contrasts the prohibitive cost of the standard CPU GPU roundtrip (moving
 * ~3.8GB of data) against the SIVF native in kernel deletion mechanism.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Header completions
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include "gist_loader.h"

using namespace faiss::gpu;

int main() {
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
    size_t nb = 1000000; 
    int nlist = 1024;
    size_t n_delete = 10000; 

    size_t d, fnb;
    // Reading the header is fast enough; alternatively, hardcoding d=960 is also valid.
    float* xb = fvecs_read(base_file, &d, &fnb);
    if(nb > fnb) nb = fnb;

    // Generate random IDs for deletion
    std::vector<faiss::idx_t> ids(nb);
    for(size_t i=0; i<nb; ++i) ids[i] = (faiss::idx_t)i;
    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    ids.resize(n_delete);
    faiss::IDSelectorBatch sel(n_delete, ids.data());

    // Reduce Temp Memory to accommodate the large GIST dataset (3.8GB resident)
    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 
    faiss::IndexFlatL2 quantizer(d);

    // ==========================================
    // Round 1: Baseline (Roundtrip Deletion)
    // ==========================================
    {
        std::cout << "[Baseline] Preparing..." << std::endl;
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        index.train(50000, xb); // Train on a subset to save time
        index.add(nb, xb);
        cudaDeviceSynchronize();

        std::cout << "[Baseline] Deleting (Roundtrip 3.8GB data!)..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // The Bottleneck: Full Index Roundtrip
        // 1. Download to CPU
        faiss::Index* cpu = faiss::gpu::index_gpu_to_cpu(&index); 
        // 2. Delete on CPU
        cpu->remove_ids(sel); 
        // 3. Re upload and Rebuild on GPU
        faiss::gpu::GpuIndexIVFFlat* new_gpu = 
            dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(faiss::gpu::index_cpu_to_gpu(&res, 0, cpu)); 
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "[Baseline] Time: " << ms << " ms" << std::endl;
        
        delete cpu; delete new_gpu;
    } 

    // Force synchronization and cleanup
    cudaDeviceSynchronize();

    // ==========================================
    // Round 2: SIVF (Native Deletion)
    // ==========================================
    {
        std::cout << "[SIVF] Preparing..." << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig cfg; cfg.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, cfg);
        
        // Critical Fix: Allocate exact capacity (1.0x) instead of 1.5x redundancy
        // GIST vectors are large, so avoiding overallocation prevents OOM.
        size_t cap = nb; 
        std::cout << "[SIVF] Allocating exact capacity: " << cap << std::endl;
        index.initSlabManager(cap, d);

        index.train(50000, xb);
        index.add(nb, xb);
        cudaDeviceSynchronize();

        std::cout << "[SIVF] Deleting (Native)..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // Native deletion, no data movement required
        index.remove_ids(sel);
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "[SIVF] Time: " << ms << " ms" << std::endl;
    }

    delete[] xb;
    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ make test_sivf_gist_delete -j
./test_sivf_gist_delete
[ 65%] Built target faiss_gpu_objs
[100%] Built target faiss
[100%] Building CXX object CMakeFiles/test_sivf_gist_delete.dir/hpdic/experiment/gist/test_sivf_gist_delete.cpp.o
[100%] Linking CXX executable test_sivf_gist_delete
[100%] Built target test_sivf_gist_delete
[Baseline] Preparing...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Deleting (Roundtrip 3.8GB data!)...
[Baseline] Time: 11842.9 ms
[SIVF] Preparing...
[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.23 s
  Iteration 19 (1.51 s, search 1.01 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Deleting (Native)...
[SIVF] Time: 0.88981 ms
cc@rtx6000:~/ElasticIVF/build$ 
 */