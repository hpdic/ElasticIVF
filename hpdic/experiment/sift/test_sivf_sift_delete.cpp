/**
 * test_sivf_sift_delete.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: SIFT1M Deletion Performance (Roundtrip vs. Native)
 *
 * This test evaluates the latency and throughput of vector deletion.
 * It compares the standard Faiss approach (which requires a costly CPU-GPU
 * synchronization and re-upload) against the SIVF native in-kernel deletion.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <random>
#include <omp.h>

// Faiss & SIVF Headers
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h> // Required for Index transfer (GPU <-> CPU)
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> 

#include "sift_loader.h"

using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Generate Random IDs for Deletion
// ---------------------------------------------------------
std::vector<faiss::idx_t> generate_delete_ids(size_t total_vectors, size_t delete_count) {
    std::vector<faiss::idx_t> ids(total_vectors);
    for(size_t i=0; i<total_vectors; ++i) ids[i] = (faiss::idx_t)i;
    
    // Shuffle to select random targets
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    
    // Keep the first N IDs
    ids.resize(delete_count);
    return ids;
}

int main(int argc, char** argv) {
    // 1. Configuration
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    size_t nb = 1000000;     // Full SIFT1M size
    int nlist = 1024;        // Number of centroids
    size_t n_delete = 10000; // Batch size (consistent with paper methodology)

    // 2. Load Data
    size_t d, file_nb;
    std::cout << "[Loader] Reading SIFT1M..." << std::endl;
    float* xb = fvecs_read(base_file, &d, &file_nb);
    if(nb > file_nb) nb = file_nb;

    // Prepare Deletion Targets
    // We use IDSelectorBatch to pass the list of IDs to remove
    std::cout << "[Prepare] Generating " << n_delete << " random IDs to delete..." << std::endl;
    std::vector<faiss::idx_t> delete_ids = generate_delete_ids(nb, n_delete);
    
    // Construct Selector
    faiss::IDSelectorBatch selector(n_delete, delete_ids.data());

    // GPU Resources
    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 512); // 512MB Temp Memory
    faiss::IndexFlatL2 quantizer(d);

    // =========================================================
    // Round 1: Baseline (CPU-GPU Roundtrip Deletion)
    // =========================================================
    {
        std::cout << "\n[Baseline] Setting up GPU Index..." << std::endl;
        faiss::gpu::GpuIndexIVFFlat gpu_index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        
        // Train & Add
        gpu_index.train(100000, xb); // Train on subset
        gpu_index.add(nb, xb);
        
        cudaDeviceSynchronize();
        std::cout << "[Baseline] Ready. Starting Deletion Benchmark (Roundtrip)..." << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();

        // --- Simulate the "Roundtrip" Workaround ---
        // Since standard GpuIndexIVF does not support in-place deletion:
        
        // 1. Download: Copy Index from GPU VRAM to CPU RAM
        faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&gpu_index);
        
        // 2. Modify: Perform deletion on the CPU
        cpu_index->remove_ids(selector);
        
        // 3. Upload: Copy Index back from CPU RAM to GPU VRAM
        // Note: This effectively rebuilds the GPU index structures
        faiss::gpu::GpuIndexIVFFlat* new_gpu_index = 
            dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index));
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  -> Time: " << time_ms << " ms" << std::endl;
        std::cout << "  -> Throughput: " << (size_t)(n_delete / (time_ms / 1000.0)) << " vecs/sec" << std::endl;

        delete cpu_index;
        delete new_gpu_index;
    }

    // =========================================================
    // Round 2: SIVF (Native In-Kernel Deletion)
    // =========================================================
    {
        std::cout << "\n[SIVF] Setting up..." << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, nlist, faiss::METRIC_L2, config);
        
        // Initialize Slab Memory Pool
        sivf_index.initSlabManager(nb * 1.5, d);

        // Train & Add
        sivf_index.train(100000, xb);
        sivf_index.add(nb, xb);

        cudaDeviceSynchronize();
        std::cout << "[SIVF] Ready. Starting Native Deletion..." << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();

        // --- Native Deletion ---
        // Executes entirely on the GPU via atomic bitmap updates.
        // No PCI-e transfer or host-side synchronization required.
        sivf_index.remove_ids(selector);

        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "  -> Time: " << time_ms << " ms" << std::endl;
        std::cout << "  -> Throughput: " << (size_t)(n_delete / (time_ms / 1000.0)) << " vecs/sec" << std::endl;
    }

    delete[] xb;
    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_sift_delete
[Loader] Reading SIFT1M...
[Prepare] Generating 10000 random IDs to delete...

[Baseline] Setting up GPU Index...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Ready. Starting Deletion Benchmark (Roundtrip)...
  -> Time: 1626.24 ms
  -> Throughput: 6149 vecs/sec

[SIVF] Setting up...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 100000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 19 (0.65 s, search 0.35 s): objective=4.83016e+09 imbalance=1.135 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Ready. Starting Native Deletion...
  -> Time: 0.856534 ms
  -> Throughput: 11674959 vecs/sec
cc@rtx6000:~/ElasticIVF/build$ 
 */