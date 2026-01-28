/**
 * File: test_sivf_mixed.cpp
 * Date: 2026-01-28
 * Description: Mixed Workload Stability Test (Insert -> Search -> Delete).
 * Demonstrates that search performance remains stable even under continuous index mutation.
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
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include "sift/sift_loader.h"

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
    const char* base_path;
    const char* query_path;
    size_t window_size = 50000;  // 50k window
    size_t batch_size = 1000;    // 1k mutation
    int steps = 1000;            // 1000 steps
    int nlist = 1024;
    int k = 10;                  // Top-k
} cfg;

int main(int argc, char** argv) {
    // 1. Config Setup
    cfg.dataset_name = "SIFT1M";
    cfg.base_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    cfg.query_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_query.fvecs";

    if (argc > 1 && std::string(argv[1]) == "gist") {
        cfg.dataset_name = "GIST1M";
        cfg.base_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
        cfg.query_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_query.fvecs";
        cfg.steps = 500;
    }

    // 2. Load Data
    size_t d, nb, nq;
    std::cout << "[Loader] Reading Base Vectors..." << std::endl;
    float* all_data = fvecs_read(cfg.base_path, &d, &nb);
    
    std::cout << "[Loader] Reading Query Vectors..." << std::endl;
    float* query_data = fvecs_read(cfg.query_path, &d, &nq);
    // Use first 100 queries for latency check (enough to measure speed, light enough to run fast)
    int num_queries = 100; 

    // Cyclic buffer logic
    size_t required_data = cfg.window_size + cfg.steps * cfg.batch_size;
    std::vector<float> workspace_data(required_data * d);
    for (size_t i = 0; i < required_data; ++i) {
        size_t src_idx = i % nb;
        std::memcpy(workspace_data.data() + i * d, all_data + src_idx * d, d * sizeof(float));
    }
    delete[] all_data;

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024);

    faiss::IndexFlatL2 quantizer(d);
    std::vector<float> train_data(50000 * d);
    std::memcpy(train_data.data(), workspace_data.data(), 50000 * d * sizeof(float));

    std::cout << "\n>>> Starting SIVF Mixed Workload Test (Insert->Search->Delete)..." << std::endl;

    faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
    faiss::gpu::GpuIndexSIVF sivf_index(&res, d, cfg.nlist, faiss::METRIC_L2, config);

    // Train & Init
    sivf_index.train(50000, train_data.data());
    cudaDeviceSynchronize();

    size_t max_possible_id = cfg.window_size + (cfg.steps * cfg.batch_size);
    sivf_index.initSlabManager(max_possible_id * 2, d);

    // Pre-fill
    std::cout << "    [Init] Pre-filling window..." << std::endl;
    std::vector<faiss::idx_t> initial_ids(cfg.window_size);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    sivf_index.add_with_ids(cfg.window_size, workspace_data.data(), initial_ids.data());

    faiss::idx_t current_max_id = cfg.window_size;
    faiss::idx_t current_min_id = 0;

    // Output buffers for search
    std::vector<float> distances(num_queries * cfg.k);
    std::vector<faiss::idx_t> labels(num_queries * cfg.k);

    LatencyStats stats_search("Search Latency (Interleaved)");

    // --- Main Loop ---
    for (int s = 0; s < cfg.steps; ++s) {
        if (s % 100 == 0) std::cout << "    Step " << s << "/" << cfg.steps << "\r" << std::flush;

        // 1. Insert
        float* batch_ptr = workspace_data.data() + (current_max_id * d);
        std::vector<faiss::idx_t> add_ids(cfg.batch_size);
        std::iota(add_ids.begin(), add_ids.end(), current_max_id);
        sivf_index.add_with_ids(cfg.batch_size, batch_ptr, add_ids.data());

        // 2. Search (The Critical Test)
        // Measure if search slows down due to the insert we just did
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        sivf_index.search(num_queries, query_data, cfg.k, distances.data(), labels.data());
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        stats_search.record(std::chrono::duration<double, std::milli>(t1 - t0).count());

        // 3. Delete
        std::vector<faiss::idx_t> del_ids(cfg.batch_size);
        std::iota(del_ids.begin(), del_ids.end(), current_min_id);
        faiss::IDSelectorBatch selector(cfg.batch_size, del_ids.data());
        sivf_index.remove_ids(selector);

        current_max_id += cfg.batch_size;
        current_min_id += cfg.batch_size;
    }

    std::cout << "\n----------------------------------------------------------" << std::endl;
    stats_search.print();
    std::cout << "----------------------------------------------------------" << std::endl;

    delete[] query_data;
    return 0;
}

/** Example output:
 * (myenv) cc@rtx6000:~/ElasticIVF$ cd ~/ElasticIVF/build
make -j test_sivf_mixed
./test_sivf_mixed
[ 64%] Built target faiss_gpu_objs
[ 97%] Built target faiss
[ 97%] Building CXX object CMakeFiles/test_sivf_mixed.dir/hpdic/experiment/test_sivf_mixed.cpp.o
[100%] Linking CXX executable test_sivf_mixed
[100%] Built target test_sivf_mixed
[Loader] Reading Base Vectors...
[Loader] Reading Query Vectors...

>>> Starting SIVF Mixed Workload Test (Insert->Search->Delete)...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.49 s, search 0.19 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 332221
  > Data Buffer: 2100000 -> 10631072 vectors (Avoids Overflow)

    [Init] Pre-filling window...
    Step 900/1000
----------------------------------------------------------
  [Search Latency (Interleaved)] Avg: 0.25 ms | P99: 0.41 ms | Max: 2.89 ms
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_mixed gist
[Loader] Reading Base Vectors...
[Loader] Reading Query Vectors...

>>> Starting SIVF Mixed Workload Test (Insert->Search->Delete)...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.14 s
  Iteration 19 (1.58 s, search 1.03 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 175971
  > Data Buffer: 1100000 -> 5631072 vectors (Avoids Overflow)

    [Init] Pre-filling window...
    Step 400/500
----------------------------------------------------------
  [Search Latency (Interleaved)] Avg: 0.69 ms | P99: 0.78 ms | Max: 0.97 ms
----------------------------------------------------------
(myenv) cc@rtx6000:~/ElasticIVF/build$ 
 */