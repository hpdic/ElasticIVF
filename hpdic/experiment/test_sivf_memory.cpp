/**
 * test_sivf_memory.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Memory Footprint & Fragmentation Benchmark (SIFT + GIST)
 *
 * This benchmark calculates the Space-Time Trade-off of SIVF across varying dataset sizes.
 * - Goal 1: Compare VRAM usage of SIVF (Slab-based) vs. Baseline (Compact Arrays).
 * - Goal 2: Verify "Zero-Cost Reclamation" by observing reuse after deletion.
 *
 * Usage:
 * ./test_sivf_memory sift   (Default)
 * ./test_sivf_memory gist
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <sstream>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include "sift/sift_loader.h"    

using namespace faiss::gpu;

// Global Configuration
struct Config {
    std::string name;
    std::string path;
    int nlist;
} cfg;

// Helper: Format number with suffix (e.g., 1000 -> "1K")
std::string format_count(size_t n) {
    if (n >= 1000000) return std::to_string(n / 1000000) + "M";
    if (n >= 1000) return std::to_string(n / 1000) + "K";
    return std::to_string(n);
}

// Helper: Calculate Theoretical Baseline Memory (MB)
// Baseline = Raw Vectors + IDs (packed tightly)
double calc_baseline_mb(size_t nb, int d) {
    double vec_data = (double)nb * d * sizeof(float);
    double idx_data = (double)nb * sizeof(long long); // ID is usually 64-bit
    return (vec_data + idx_data) / (1024.0 * 1024.0);
}

// Helper: Calculate SIVF Memory (MB)
double calc_sivf_mb(size_t capacity, int d) {
    // SIVF Architecture Constants
    const size_t VECS_PER_SLAB = 32;
    const size_t SLAB_HEADER_BYTES = 128; // Header + Padding + Bitmap
    
    // Size of one slot (Vector + ID)
    size_t slot_size = d * sizeof(float) + sizeof(long long);
    
    // Size of one full Slab
    size_t slab_size = SLAB_HEADER_BYTES + VECS_PER_SLAB * slot_size;
    
    // Total Slabs Allocated
    // Capacity is rounded up to the nearest multiple of 32
    size_t total_slabs = (capacity + VECS_PER_SLAB - 1) / VECS_PER_SLAB;
    
    return (double)(total_slabs * slab_size) / (1024.0 * 1024.0);
}

struct ResultRow {
    size_t count;
    double base_mb;
    double sivf_mb;
    double overhead;
    bool reuse_success;
};

int main(int argc, char** argv) {
    // 1. Argument Parsing & Configuration
    std::string target = "sift";
    if (argc > 1) target = std::string(argv[1]);

    // Test points: 100K, 200K, 500K, 1M (Max for GIST is ~3.8GB, safe for 24GB VRAM)
    std::vector<size_t> test_counts = {100000, 200000, 500000, 1000000}; 

    if (target == "gist") {
        cfg.name = "GIST1M";
        cfg.path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
        cfg.nlist = 1024;
    } else {
        cfg.name = "SIFT1M";
        cfg.path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
        cfg.nlist = 1024;
    }

    // 2. Load Data (Load max needed upfront)
    size_t max_count = test_counts.back();
    size_t d, fnb;
    std::cout << "[Loader] Reading " << cfg.name << " from " << cfg.path << "..." << std::endl;
    float* all_data = fvecs_read(cfg.path.c_str(), &d, &fnb);
    
    if (max_count > fnb) {
        std::cerr << "[Error] Requested " << max_count << " vectors but file only has " << fnb << std::endl;
        return -1;
    }

    // Create a copy for training (first 50K)
    size_t n_train = 50000;
    std::vector<float> xt(n_train * d);
    std::memcpy(xt.data(), all_data, n_train * d * sizeof(float));

    // 3. Setup GPU Resources
    StandardGpuResources res;
    if (d > 500) res.setTempMemory(512 * 1024 * 1024); 
    else res.setTempMemory(1024 * 1024 * 1024);

    faiss::IndexFlatL2 quantizer(d);
    std::vector<ResultRow> results;

    std::cout << "\n=== Starting Memory Benchmark Loop (" << cfg.name << ") ===" << std::endl;

    for (size_t count : test_counts) {
        std::cout << "\n>>> Testing Size: " << format_count(count) << " (" << count << ")" << std::endl;
        
        // --- Baseline Calculation ---
        double base_mb = calc_baseline_mb(count, d);

        // --- SIVF Measurement ---
        bool reuse_success = false;
        double sivf_mb = 0.0;
        
        try {
            faiss::gpu::GpuIndexIVFFlatConfig config; 
            config.device = 0;
            faiss::gpu::GpuIndexSIVF sivf(&res, d, cfg.nlist, faiss::METRIC_L2, config);
            
            // Allocate EXACT capacity
            sivf.initSlabManager(count, d); 
            
            // Train and Add
            sivf.train(n_train, xt.data());
            sivf.add(count, all_data); // Use the pointer to the large buffer, take first 'count'

            sivf_mb = calc_sivf_mb(count, d);

            // --- Reuse Verification (Mini Test) ---
            // Delete 10% and re-insert to verify stability
            size_t n_del = count / 10;
            std::vector<faiss::idx_t> del_ids(n_del);
            for(size_t i=0; i<n_del; ++i) del_ids[i] = (faiss::idx_t)i; 
            faiss::IDSelectorBatch sel(n_del, del_ids.data());
            
            sivf.remove_ids(sel);
            cudaDeviceSynchronize();
            
            // Re-insert same data
            sivf.add(n_del, all_data);
            cudaDeviceSynchronize();
            
            reuse_success = true; // If we reached here without OOM/Exception
            
        } catch (const std::exception& e) {
            std::cerr << "   [FAIL] Exception: " << e.what() << std::endl;
            reuse_success = false;
        }

        double overhead = (sivf_mb / base_mb - 1.0) * 100.0;
        results.push_back({count, base_mb, sivf_mb, overhead, reuse_success});
        
        std::cout << "   [Done] Overhead: " << std::fixed << std::setprecision(2) << overhead << "%" << std::endl;
    }

    delete[] all_data;

    // 4. Print Summary Table for Plotting
    std::cout << "\n========================================================" << std::endl;
    std::cout << " SUMMARY: Memory Overhead Analysis (" << cfg.name << ")" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << std::left << std::setw(10) << "Count" 
              << "| " << std::setw(15) << "Baseline(MB)" 
              << "| " << std::setw(15) << "SIVF(MB)" 
              << "| " << std::setw(12) << "Overhead(%)" 
              << "| " << "Reuse?" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(10) << format_count(r.count)
                  << "| " << std::setw(15) << std::fixed << std::setprecision(2) << r.base_mb
                  << "| " << std::setw(15) << r.sivf_mb
                  << "| " << std::setw(12) << r.overhead
                  << "| " << (r.reuse_success ? "YES" : "NO") << std::endl;
    }
    std::cout << "========================================================" << std::endl;

    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_memory # default is SIFT
./test_sivf_memory gist
[Loader] Reading SIFT1M from /home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs...

=== Starting Memory Benchmark Loop (SIFT1M) ===

>>> Testing Size: 100K (100000)
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 19721
  > Data Buffer: 100000 -> 631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.27 s, search 0.19 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.77%

>>> Testing Size: 200K (200000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 35346
  > Data Buffer: 200000 -> 1131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.23 s, search 0.16 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.77%

>>> Testing Size: 500K (500000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 82221
  > Data Buffer: 500000 -> 2631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.23 s, search 0.16 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.77%

>>> Testing Size: 1M (1000000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 19 (0.23 s, search 0.15 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.77%

========================================================
 SUMMARY: Memory Overhead Analysis (SIFT1M)
========================================================
Count     | Baseline(MB)   | SIVF(MB)       | Overhead(%) | Reuse?
--------------------------------------------------------
100K      | 49.59          | 49.97          | 0.77        | YES
200K      | 99.18          | 99.95          | 0.77        | YES
500K      | 247.96         | 249.86         | 0.77        | YES
1M        | 495.91         | 499.73         | 0.77        | YES
========================================================
[Loader] Reading GIST1M from /home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs...

=== Starting Memory Benchmark Loop (GIST1M) ===

>>> Testing Size: 100K (100000)
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 19721
  > Data Buffer: 100000 -> 631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.16 s
  Iteration 19 (1.52 s, search 1.06 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.10%

>>> Testing Size: 200K (200000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 35346
  > Data Buffer: 200000 -> 1131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.16 s
  Iteration 19 (1.47 s, search 1.00 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.10%

>>> Testing Size: 500K (500000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 82221
  > Data Buffer: 500000 -> 2631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.16 s
  Iteration 19 (1.48 s, search 1.00 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.10%

>>> Testing Size: 1M (1000000)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.16 s
  Iteration 19 (1.56 s, search 1.07 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
   [Done] Overhead: 0.10%

========================================================
 SUMMARY: Memory Overhead Analysis (GIST1M)
========================================================
Count     | Baseline(MB)   | SIVF(MB)       | Overhead(%) | Reuse?
--------------------------------------------------------
100K      | 366.97         | 367.36         | 0.10        | YES
200K      | 733.95         | 734.71         | 0.10        | YES
500K      | 1834.87        | 1836.78        | 0.10        | YES
1M        | 3669.74        | 3673.55        | 0.10        | YES
========================================================
cc@rtx6000:~/ElasticIVF/build$ 
 */