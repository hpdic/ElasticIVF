/**
 * test_sivf_sliding.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Real-world Streaming Benchmark: Sliding Window (FIFO)
 *
 * This test simulates a real-time data streaming scenario where the index
 * maintains a fixed window size (e.g., 200k vectors). As new data arrives,
 * old data is evicted to keep the memory footprint stable.
 *
 * Workflow:
 * 1. Initialize an index with a pre-filled window.
 * 2. Continuous Loop:
 * a. Ingest a new batch of vectors (Insertion).
 * b. Evict the oldest batch of vectors (Deletion).
 *
 * Comparison:
 * - SIVF: Executes native in-place Insertion and Deletion on the GPU.
 * - Faiss Baseline: Lacks native GPU deletion support; requires a costly
 * CPU Roundtrip (Download GPU Index -> CPU Delete -> Re-upload to GPU).
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h> // Required for Index transfer (GPU <-> CPU)
#include <faiss/gpu/GpuIndexSIVF.h>
#include "sift/sift_loader.h"

using namespace faiss::gpu;

// Global Configuration Structure
struct Config {
    std::string dataset_name;
    const char* file_path;
    size_t window_size = 200000; // Number of vectors resident in VRAM
    size_t batch_size = 10000;   // Update step size (Insert/Delete count)
    int steps = 10;              // Number of sliding iterations
    int nlist = 1024;            // Number of inverted lists (centroids)
} cfg;

int main(int argc, char** argv) {
    // Default to SIFT1M; allow switching to GIST1M via command line argument
    cfg.dataset_name = "SIFT1M";
    cfg.file_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";

    if (argc > 1 && std::string(argv[1]) == "gist") {
        cfg.dataset_name = "GIST1M";
        cfg.file_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
        cfg.nlist = 1024; 
        // GIST vectors are high-dimensional (960d). Reduce window size to prevent OOM.
        cfg.window_size = 100000; 
        cfg.batch_size = 5000;
    }

    // 1. Data Loading
    size_t d, total_nb;
    std::cout << "[Loader] Reading " << cfg.dataset_name << "..." << std::endl;
    float* all_data = fvecs_read(cfg.file_path, &d, &total_nb);
    
    // Ensure sufficient data for the simulation (Cyclic tiling if necessary)
    size_t required_data = cfg.window_size + cfg.steps * cfg.batch_size;
    std::vector<float> workspace_data;
    if (total_nb < required_data) {
        std::cout << "[Warn] Dataset smaller than required simulation length. Tiling data..." << std::endl;
        workspace_data.resize(required_data * d);
        for (size_t i = 0; i < required_data; ++i) {
            size_t src_idx = i % total_nb;
            std::memcpy(workspace_data.data() + i * d, all_data + src_idx * d, d * sizeof(float));
        }
    } else {
        // Copy the required segment to workspace memory
        workspace_data.resize(required_data * d);
        std::memcpy(workspace_data.data(), all_data, required_data * d * sizeof(float));
    }
    delete[] all_data; // Release original raw buffer

    // 2. Resource Initialization
    StandardGpuResources res;
    // Adjust temporary memory limit based on dimensionality
    if (d > 500) res.setTempMemory(512 * 1024 * 1024); // Restrict Temp for GIST
    else res.setTempMemory(1024 * 1024 * 1024);        // Allocate 1GB for SIFT

    faiss::IndexFlatL2 quantizer(d);
    
    // Prepare Training Data (Use the first 50k vectors)
    std::vector<float> train_data(50000 * d);
    std::memcpy(train_data.data(), workspace_data.data(), 50000 * d * sizeof(float));

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " Benchmark: Sliding Window (" << cfg.dataset_name << ")" << std::endl;
    std::cout << " Window Size: " << cfg.window_size << " | Batch Size: " << cfg.batch_size << std::endl;
    std::cout << " Steps: " << cfg.steps << " | Dim: " << d << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // =========================================================
    // Round 1: Faiss Baseline (CPU Roundtrip Deletion)
    // =========================================================
    {
        std::cout << ">>> Running Baseline (Faiss GPU)..." << std::endl;
        
        // Initialize & Train
        faiss::gpu::GpuIndexIVFFlat* gpu_index = new faiss::gpu::GpuIndexIVFFlat(&res, &quantizer, d, cfg.nlist, faiss::METRIC_L2);
        gpu_index->train(50000, train_data.data());

        // Pre-fill Window
        std::cout << "    [Init] Pre-filling window (" << cfg.window_size << " vecs)..." << std::endl;
        // Generate sequential IDs starting from 0
        std::vector<faiss::idx_t> initial_ids(cfg.window_size);
        std::iota(initial_ids.begin(), initial_ids.end(), 0);
        gpu_index->add_with_ids(cfg.window_size, workspace_data.data(), initial_ids.data());

        faiss::idx_t current_max_id = cfg.window_size;
        faiss::idx_t current_min_id = 0; // ID of the oldest data

        printf("\n| %-4s | %-12s | %-12s | %-12s |\n", "Step", "Add(ms)", "Del(ms)", "Total(ms)");
        printf("|------|--------------|--------------|--------------|\n");

        for (int s = 0; s < cfg.steps; ++s) {
            // 1. Prepare Data Batches
            float* batch_ptr = workspace_data.data() + (current_max_id * d);
            std::vector<faiss::idx_t> add_ids(cfg.batch_size);
            std::iota(add_ids.begin(), add_ids.end(), current_max_id);

            std::vector<faiss::idx_t> del_ids(cfg.batch_size);
            std::iota(del_ids.begin(), del_ids.end(), current_min_id);
            faiss::IDSelectorBatch selector(cfg.batch_size, del_ids.data());

            // 2. Benchmark Insertion
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            gpu_index->add_with_ids(cfg.batch_size, batch_ptr, add_ids.data());
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double t_add = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // 3. Benchmark Deletion (The Expensive Roundtrip)
            auto t2 = std::chrono::high_resolution_clock::now();
            
            // (A) Download Index to CPU
            faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(gpu_index);
            // (B) Execute Deletion on CPU
            cpu_index->remove_ids(selector);
            // (C) Upload back to GPU (Rebuilds the GPU index)
            delete gpu_index; // Destroy old instance
            gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(
                faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index));
            delete cpu_index; // Destroy temporary CPU index

            cudaDeviceSynchronize();
            auto t3 = std::chrono::high_resolution_clock::now();
            double t_del = std::chrono::duration<double, std::milli>(t3 - t2).count();

            printf("| %-4d | %-12.2f | %-12.2f | %-12.2f |\n", s, t_add, t_del, t_add + t_del);

            // Update sliding window cursors
            current_max_id += cfg.batch_size;
            current_min_id += cfg.batch_size;
        }
        delete gpu_index;
    }

    // =========================================================
    // Round 2: SIVF (Native In-Place Operations)
    // =========================================================
    {
        std::cout << "\n>>> Running SIVF (Ours)..." << std::endl;

        faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, cfg.nlist, faiss::METRIC_L2, config);

        // Initialize Memory Pool
        // Advantage: SIVF supports stable memory usage. We only need capacity for 
        // the active window plus a small buffer for the batch update.
        size_t cap = cfg.window_size + cfg.batch_size * 2; 
        sivf_index.initSlabManager(cap, d);

        sivf_index.train(50000, train_data.data());

        // Pre-fill Window
        std::cout << "    [Init] Pre-filling window..." << std::endl;
        std::vector<faiss::idx_t> initial_ids(cfg.window_size);
        std::iota(initial_ids.begin(), initial_ids.end(), 0);
        sivf_index.add_with_ids(cfg.window_size, workspace_data.data(), initial_ids.data());

        faiss::idx_t current_max_id = cfg.window_size;
        faiss::idx_t current_min_id = 0;

        printf("\n| %-4s | %-12s | %-12s | %-12s |\n", "Step", "Add(ms)", "Del(ms)", "Total(ms)");
        printf("|------|--------------|--------------|--------------|\n");

        for (int s = 0; s < cfg.steps; ++s) {
            // 1. Prepare Data Batches
            float* batch_ptr = workspace_data.data() + (current_max_id * d);
            std::vector<faiss::idx_t> add_ids(cfg.batch_size);
            std::iota(add_ids.begin(), add_ids.end(), current_max_id);

            std::vector<faiss::idx_t> del_ids(cfg.batch_size);
            std::iota(del_ids.begin(), del_ids.end(), current_min_id);
            faiss::IDSelectorBatch selector(cfg.batch_size, del_ids.data());

            // 2. Benchmark Insertion (Native)
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            sivf_index.add_with_ids(cfg.batch_size, batch_ptr, add_ids.data());
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double t_add = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // 3. Benchmark Deletion (Native)
            auto t2 = std::chrono::high_resolution_clock::now();
            
            // SIVF executes deletion via a single GPU kernel call (bitmap update)
            sivf_index.remove_ids(selector);

            cudaDeviceSynchronize();
            auto t3 = std::chrono::high_resolution_clock::now();
            double t_del = std::chrono::duration<double, std::milli>(t3 - t2).count();

            printf("| %-4d | %-12.2f | %-12.2f | %-12.2f |\n", s, t_add, t_del, t_add + t_del);

            current_max_id += cfg.batch_size;
            current_min_id += cfg.batch_size;
        }
    }

    return 0;
}

/**
 * Example output:
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_sliding # default is SIFT
./test_sivf_sliding gist
[Loader] Reading SIFT1M...

==========================================================
 Benchmark: Sliding Window (SIFT1M)
 Window Size: 200000 | Batch Size: 10000
 Steps: 10 | Dim: 128
==========================================================

>>> Running Baseline (Faiss GPU)...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
    [Init] Pre-filling window (200000 vecs)...

| Step | Add(ms)      | Del(ms)      | Total(ms)    |
|------|--------------|--------------|--------------|
| 0    | 289.00       | 368.33       | 657.33       |
| 1    | 34.52        | 338.31       | 372.83       |
| 2    | 29.05        | 325.66       | 354.71       |
| 3    | 29.59        | 329.44       | 359.04       |
| 4    | 29.80        | 324.98       | 354.78       |
| 5    | 29.86        | 328.86       | 358.73       |
| 6    | 30.38        | 328.33       | 358.71       |
| 7    | 29.41        | 324.95       | 354.36       |
| 8    | 30.17        | 327.32       | 357.50       |
| 9    | 29.23        | 333.16       | 362.39       |

>>> Running SIVF (Ours)...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 38471
  > Data Buffer: 220000 -> 1231072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.46 s, search 0.17 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
    [Init] Pre-filling window...

| Step | Add(ms)      | Del(ms)      | Total(ms)    |
|------|--------------|--------------|--------------|
| 0    | 1.83         | 0.82         | 2.65         |
| 1    | 1.84         | 0.45         | 2.29         |
| 2    | 1.87         | 0.39         | 2.26         |
| 3    | 1.81         | 0.39         | 2.20         |
| 4    | 1.84         | 0.37         | 2.22         |
| 5    | 1.83         | 0.38         | 2.21         |
| 6    | 1.78         | 0.37         | 2.15         |
| 7    | 1.81         | 0.36         | 2.17         |
| 8    | 1.79         | 0.38         | 2.17         |
| 9    | 1.82         | 0.38         | 2.20         |
[Loader] Reading GIST1M...

==========================================================
 Benchmark: Sliding Window (GIST1M)
 Window Size: 100000 | Batch Size: 5000
 Steps: 10 | Dim: 960
==========================================================

>>> Running Baseline (Faiss GPU)...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
    [Init] Pre-filling window (100000 vecs)...

| Step | Add(ms)      | Del(ms)      | Total(ms)    |
|------|--------------|--------------|--------------|
| 0    | 241.75       | 1287.21      | 1528.96      |
| 1    | 53.50        | 1094.06      | 1147.55      |
| 2    | 49.23        | 1079.81      | 1129.04      |
| 3    | 46.96        | 1079.22      | 1126.18      |
| 4    | 45.53        | 1084.52      | 1130.05      |
| 5    | 45.77        | 1078.11      | 1123.88      |
| 6    | 45.09        | 1076.43      | 1121.52      |
| 7    | 45.00        | 1074.14      | 1119.14      |
| 8    | 45.28        | 1071.68      | 1116.96      |
| 9    | 45.64        | 1075.57      | 1121.21      |

>>> Running SIVF (Ours)...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 21286
  > Data Buffer: 110000 -> 681152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.22 s
  Iteration 19 (1.22 s, search 0.73 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
    [Init] Pre-filling window...

| Step | Add(ms)      | Del(ms)      | Total(ms)    |
|------|--------------|--------------|--------------|
| 0    | 3.89         | 0.69         | 4.58         |
| 1    | 4.20         | 0.28         | 4.48         |
| 2    | 4.30         | 0.27         | 4.57         |
| 3    | 4.16         | 0.26         | 4.42         |
| 4    | 4.26         | 0.25         | 4.51         |
| 5    | 3.95         | 0.25         | 4.20         |
| 6    | 3.87         | 0.24         | 4.11         |
| 7    | 3.87         | 0.24         | 4.11         |
| 8    | 3.90         | 0.24         | 4.14         |
| 9    | 3.86         | 0.24         | 4.10         |
cc@rtx6000:~/ElasticIVF/build$ 
 */