/**
 * File: test_sivf_p99.cpp
 * Date: 2026-01-28
 * Description: Stability Benchmark for SIVF (P99 Analysis).
 * Uses a "Lightweight Long-Run" configuration to prevent VRAM overflow/ID-bound errors
 * while providing rigorous tail latency statistics.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <iomanip>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include "sift/sift_loader.h"
#include <faiss/gpu/GpuIndexIVFFlat.h>

using namespace faiss::gpu;

// --- Stats Helper ---
struct LatencyStats {
    std::vector<double> records;
    std::string name;
    LatencyStats(std::string n) : name(n) { records.reserve(10000); }
    void record(double ms) { records.push_back(ms); }
    void print() {
        if (records.empty()) return;
        std::sort(records.begin(), records.end());
        double sum = std::accumulate(records.begin(), records.end(), 0.0);
        double avg = sum / records.size();
        double p99 = records[(size_t)(records.size() * 0.99)];
        double max = records.back();
        std::cout << "  [" << name << "] Avg: " << std::fixed << std::setprecision(2) << avg 
                  << " ms | \033[1;32mP99: " << p99 << " ms\033[0m | Max: " << max << " ms" << std::endl;
    }
};

struct Config {
    std::string dataset_name;
    const char* file_path;
    // [Small Scale Config] for Stability Testing
    size_t window_size = 50000;  // 50k vectors
    size_t batch_size = 1000;    // 1k updates per step
    int steps = 1000;            // 1000 steps (Total IDs = 50k + 1M = ~1.05M)
    int nlist = 1024;
} cfg;

int main(int argc, char** argv) {
    cfg.dataset_name = "SIFT1M";
    cfg.file_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";

    if (argc > 1 && std::string(argv[1]) == "gist") {
        cfg.dataset_name = "GIST1M";
        cfg.file_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
        cfg.steps = 500; 
    }

    // 1. Load Data
    size_t d, total_nb;
    std::cout << "[Loader] Reading " << cfg.dataset_name << " (Lightweight Mode)..." << std::endl;
    float* all_data = fvecs_read(cfg.file_path, &d, &total_nb);
    
    // Cyclic buffer logic
    size_t required_data = cfg.window_size + cfg.steps * cfg.batch_size;
    std::vector<float> workspace_data(required_data * d);
    for (size_t i = 0; i < required_data; ++i) {
        size_t src_idx = i % total_nb;
        std::memcpy(workspace_data.data() + i * d, all_data + src_idx * d, d * sizeof(float));
    }
    delete[] all_data;

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp

    faiss::IndexFlatL2 quantizer(d);
    std::vector<float> train_data(50000 * d);
    std::memcpy(train_data.data(), workspace_data.data(), 50000 * d * sizeof(float));

    std::cout << "\n>>> Starting SIVF P99 Test (Window=" << cfg.window_size 
              << ", Batch=" << cfg.batch_size << ")..." << std::endl;

    faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
    faiss::gpu::GpuIndexSIVF sivf_index(&res, d, cfg.nlist, faiss::METRIC_L2, config);

    // [Step 1] Train FIRST
    sivf_index.train(50000, train_data.data());
    cudaDeviceSynchronize();

    // [Step 2] Calc Max IDs & Init
    size_t max_possible_id = cfg.window_size + (cfg.steps * cfg.batch_size);
    size_t safe_cap = max_possible_id * 2; 
    
    std::cout << "    [Alloc] Max ID expected: " << max_possible_id 
              << " -> Allocating Table for: " << safe_cap << std::endl;
    sivf_index.initSlabManager(safe_cap, d);

    // [Step 3] Pre-fill
    std::cout << "    [Init] Pre-filling window..." << std::endl;
    std::vector<faiss::idx_t> initial_ids(cfg.window_size);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    sivf_index.add_with_ids(cfg.window_size, workspace_data.data(), initial_ids.data());

    faiss::idx_t current_max_id = cfg.window_size;
    faiss::idx_t current_min_id = 0;

    LatencyStats stats_del("SIVF Deletion P99");

    for (int s = 0; s < cfg.steps; ++s) {
        if (s % 100 == 0) std::cout << "    Step " << s << "/" << cfg.steps << "\r" << std::flush;

        float* batch_ptr = workspace_data.data() + (current_max_id * d);
        std::vector<faiss::idx_t> add_ids(cfg.batch_size);
        std::iota(add_ids.begin(), add_ids.end(), current_max_id);

        std::vector<faiss::idx_t> del_ids(cfg.batch_size);
        std::iota(del_ids.begin(), del_ids.end(), current_min_id);
        faiss::IDSelectorBatch selector(cfg.batch_size, del_ids.data());

        // Insert (Don't measure, just do it)
        sivf_index.add_with_ids(cfg.batch_size, batch_ptr, add_ids.data());
        
        // Measure Deletion (Stability Critical Path)
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        sivf_index.remove_ids(selector);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        stats_del.record(std::chrono::duration<double, std::milli>(t1 - t0).count());

        current_max_id += cfg.batch_size;
        current_min_id += cfg.batch_size;
    }
    
    std::cout << "\n----------------------------------------------------------" << std::endl;
    stats_del.print();
    std::cout << "----------------------------------------------------------" << std::endl;

    return 0;
}

/** Example output:
 * (myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_p99
[Loader] Reading SIFT1M (Lightweight Mode)...

>>> Starting SIVF P99 Test (Window=50000, Batch=1000)...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.28 s, search 0.19 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
    [Alloc] Max ID expected: 1050000 -> Allocating Table for: 2100000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 332221
  > Data Buffer: 2100000 -> 10631072 vectors (Avoids Overflow)

    [Init] Pre-filling window...
    Step 900/1000
----------------------------------------------------------
  [SIVF Deletion P99] Avg: 0.08 ms | P99: 0.10 ms | Max: 0.58 ms
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_p99 gist
[Loader] Reading GIST1M (Lightweight Mode)...

>>> Starting SIVF P99 Test (Window=50000, Batch=1000)...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.14 s
  Iteration 19 (1.52 s, search 1.04 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
    [Alloc] Max ID expected: 550000 -> Allocating Table for: 1100000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 175971
  > Data Buffer: 1100000 -> 5631072 vectors (Avoids Overflow)

    [Init] Pre-filling window...
    Step 400/500
----------------------------------------------------------
  [SIVF Deletion P99] Avg: 0.08 ms | P99: 0.10 ms | Max: 0.53 ms
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */