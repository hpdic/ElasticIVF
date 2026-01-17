/**
 * test_sivf_t2i_delete.cpp
 * 
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: T2I Deletion Performance (Roundtrip Baseline vs. Native SIVF)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include <faiss/gpu/GpuCloner.h> 
#include <faiss/IndexFlat.h>
#include "t2i_loader.h" 

using namespace faiss::gpu;

int main() {
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/t2i/t2i_base_1M.fbin";
    size_t nb_load = 1000000;
    int nlist = 1024;
    size_t delete_batch_size = 10000; // Delete 10k vectors

    size_t d, nb;
    std::cout << "[Loader] Reading T2I Base..." << std::endl;
    float* xb = fbin_read(base_file, &d, &nb);
    if(nb_load > nb) nb_load = nb;

    StandardGpuResources res;
    res.setTempMemory(1024L * 1024 * 1024);

    // Prepare Delete IDs (First 10k IDs)
    std::vector<faiss::idx_t> del_ids(delete_batch_size);
    std::iota(del_ids.begin(), del_ids.end(), 0);
    faiss::IDSelectorBatch sel(delete_batch_size, del_ids.data());

    // ==========================================
    // Round 1: Baseline Deletion (The Expensive Roundtrip)
    // ==========================================
    {
        faiss::IndexFlatL2 quantizer(d);
        faiss::gpu::GpuIndexIVFFlat baseline(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        baseline.train(50000, xb);
        baseline.add(nb_load, xb);
        cudaDeviceSynchronize();

        std::cout << "[Baseline] Deleting " << delete_batch_size << " vectors via CPU Roundtrip..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // --- STEP 1: GPU -> CPU ---
        faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&baseline);
        
        // --- STEP 2: Delete on CPU ---
        long n_removed = cpu_index->remove_ids(sel);
        
        // --- STEP 3: CPU -> GPU (Re-upload) ---
        faiss::gpu::GpuIndexIVFFlat* new_gpu_index = 
            dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index));
            
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  -> Time: " << ms << " ms (Removed: " << n_removed << ")" << std::endl;

        delete cpu_index;
        delete new_gpu_index;
    }

    // ==========================================
    // Round 2: SIVF Deletion (Native In-Place)
    // ==========================================
    {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf(&res, d, nlist, faiss::METRIC_L2, config);

        sivf.initSlabManager(nb_load * 1.1, d);
        sivf.train(50000, xb);
        sivf.add(nb_load, xb);
        cudaDeviceSynchronize();

        std::cout << "[SIVF] Deleting " << delete_batch_size << " vectors in-place..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // SIVF supports native GPU deletion
        long n_removed = sivf.remove_ids(sel);
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  -> Time: " << ms << " ms (Removed: " << n_removed << ")" << std::endl;
    }

    delete[] xb;
    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_t2i_delete
[Loader] Reading T2I Base...
[Loader] Header info -> N: 1000000, D: 200
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Deleting 10000 vectors via CPU Roundtrip...
  -> Time: 2416.17 ms (Removed: 10000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200 -> 175971
  > Data Buffer: 1100000 -> 5631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 200D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.05 s
  Iteration 19 (0.33 s, search 0.25 s): objective=22642.1 imbalance=1.206 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Deleting 10000 vectors in-place...
  -> Time: 0.872174 ms (Removed: 10000)
 */