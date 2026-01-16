/**
 * faiss/hpdic/experiment/test_sivf_delete.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark: SIVF Deletion Performance
 *
 * This test measures the latency of batch deletion operations in SIVF and
 * compares it against a hardcoded Baseline (Vanilla Faiss) value to report
 * the speedup factor.
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/IDSelector.h>

// Helper function: Generate random vectors
void generate_data(size_t n, int d, std::vector<float>& data) {
    for (size_t i = 0; i < n * d; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main() {
    // ==========================================
    // Core Parameter Configuration
    // ==========================================
    int d = 128;
    int nlist = 4096;
    size_t nb = 1000000; // 1M Database vectors

    // [Modification] Set to 10000 to align with Baseline configuration
    size_t n_delete = 10000;

    // Baseline data (used for direct speedup calculation)
    // Value derived from standard Faiss GPU IVFFlat deletion benchmark
    double baseline_time_ms = 202.2;

    // ==========================================

    // 1. Prepare Data
    std::vector<float> xb(nb * d);
    std::cout << "Generating " << nb << " vectors..." << std::endl;
    generate_data(nb, d, xb);

    std::vector<faiss::idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i)
        ids[i] = (faiss::idx_t)i;

    // 2. Initialize SIVF
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2);

    // Pre allocate memory pool
    size_t max_vecs = (size_t)(nb * 1.5);
    // Allocate extra buffer for metadata
    index.initSlabManager(max_vecs, max_vecs / 32 + 20000);

    // 3. Train and Insert
    std::cout << "Training..." << std::endl;
    // Use a subset for training
    index.train(50000, xb.data());

    std::cout << "Adding " << nb << " vectors..." << std::endl;
    index.add_with_ids(nb, xb.data(), ids.data());

    cudaDeviceSynchronize();

    // 4. Prepare deletion list (delete first n_delete vectors)
    std::vector<faiss::idx_t> ids_to_delete;
    ids_to_delete.reserve(n_delete);
    for (size_t i = 0; i < n_delete; ++i) {
        ids_to_delete.push_back(ids[i]);
    }

    faiss::IDSelectorBatch selector(n_delete, ids_to_delete.data());

    // 5. Benchmark: Execute Deletion
    std::cout << "Benchmark: Removing " << n_delete << " vectors..."
              << std::endl;

    // GPU Warmup (Optional)
    // cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    // Core invocation
    size_t removed_count = index.remove_ids(selector);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 6. Print Performance Report
    std::cout << "\n================ PERFORMANCE REPORT ================"
              << std::endl;
    std::cout << "Workload:        Delete " << n_delete << " vectors"
              << std::endl;
    std::cout << "Baseline (Faiss):" << baseline_time_ms << " ms" << std::endl;
    std::cout << "SIVF (Ours):     " << ms << " ms" << std::endl;
    std::cout << "----------------------------------------------------"
              << std::endl;
    std::cout << "Speedup:         " << baseline_time_ms / ms << " x"
              << std::endl;
    std::cout << "Latency per ID:  " << (ms / n_delete) * 1000 << " ns"
              << std::endl;
    std::cout << "Actual Removed:  " << removed_count
              << " (Recall Rate: " << (double)removed_count / n_delete * 100.0
              << "%)" << std::endl;
    std::cout << "====================================================\n"
              << std::endl;

    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ make test_sivf_delete -j
[ 65%] Built target faiss_gpu_objs
[100%] Built target faiss
[100%] Building CXX object faiss/gpu/CMakeFiles/test_sivf_delete.dir/__/__/hpdic/experiment/test_sivf_delete.cpp.o
[100%] Linking CXX executable test_sivf_delete
[100%] Built target test_sivf_delete
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_delete 
Generating 1000000 vectors...
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66875 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 50000 points to 4096 centroids: please provide at least 159744 training points
Clustering 50000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.39 s, search 0.26 s): objective=420117 imbalance=1.874 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Adding 1000000 vectors...
Benchmark: Removing 10000 vectors...

================ PERFORMANCE REPORT ================
Workload:        Delete 10000 vectors
Baseline (Faiss):202.2 ms
SIVF (Ours):     0.681474 ms
----------------------------------------------------
Speedup:         296.71 x
Latency per ID:  0.0681474 ns
Actual Removed:  10000 (Recall Rate: 100%)
====================================================

cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_delete 
Generating 1000000 vectors...
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66875 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 50000 points to 4096 centroids: please provide at least 159744 training points
Clustering 50000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.38 s, search 0.27 s): objective=420117 imbalance=1.874 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Adding 1000000 vectors...
Benchmark: Removing 10000 vectors...

================ PERFORMANCE REPORT ================
Workload:        Delete 10000 vectors
Baseline (Faiss):202.2 ms
SIVF (Ours):     0.683111 ms
----------------------------------------------------
Speedup:         295.999 x
Latency per ID:  0.0683111 ns
Actual Removed:  10000 (Recall Rate: 100%)
====================================================

cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_delete 
Generating 1000000 vectors...
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66875 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 50000 points to 4096 centroids: please provide at least 159744 training points
Clustering 50000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.39 s, search 0.27 s): objective=420117 imbalance=1.874 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Adding 1000000 vectors...
Benchmark: Removing 10000 vectors...

================ PERFORMANCE REPORT ================
Workload:        Delete 10000 vectors
Baseline (Faiss):202.2 ms
SIVF (Ours):     0.66785 ms
----------------------------------------------------
Speedup:         302.763 x
Latency per ID:  0.066785 ns
Actual Removed:  10000 (Recall Rate: 100%)
====================================================

cc@rtx6000:~/ElasticIVF/build$ 
 */