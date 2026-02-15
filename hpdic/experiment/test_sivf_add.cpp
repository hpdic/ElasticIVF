/**
 * File: faiss/hpdic/experiment/test_sivf_add.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Comprehensive Benchmark: SIVF vs Vanilla Faiss (IVFFlat)
 * Parameter Sweep: nb (Database Size) x nlist (Cluster Count)
 *
 * This test evaluates the ingestion throughput (vectors/second) of the Slab-based
 * architecture against the standard contiguous memory implementation.
 *
 * Last Update: 2026-2-14
 */

#include <sys/time.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstring> // For memcpy

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h> // HPDIC SIVF Header
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss;
using namespace faiss::gpu;

// High precision timer
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Fast random data generation
// Replicates a small chunk of random data to avoid the high overhead of 
// calling RNG for millions of vectors.
void generate_data(size_t n, int d, std::vector<float>& data) {
    size_t chunk = std::min(n, (size_t)10000);
    
    // Generate the seed chunk
    for (size_t i = 0; i < chunk * d; ++i) {
        data[i] = (float)drand48();
    }
    
    // Replicate the chunk
    for (size_t i = chunk; i < n; ++i) {
        memcpy(data.data() + i * d,
               data.data() + (i % chunk) * d,
               d * sizeof(float));
    }
}

int main() {
    // ==========================================
    // Experiment Configuration (Parameter Sweep)
    // ==========================================
    int d = 128;

    // Sweep: Cluster Counts (nlist)
    std::vector<int> nlist_list = {1024, 2048, 4096};

    // Sweep: Database Sizes (1M to 4M vectors)
    // Note: 10M * 128 * 4B equals approx 5GB VRAM, fitting easily on RTX 6000
    std::vector<size_t> nb_list = {1000000, 2000000, 4000000};

    // Max capacity for pre allocation
    size_t max_nb = 10000000;
    size_t max_nt = 256 * 1024; // Sufficient training set size

    printf("Preparing Data (Max NB=%ld, Max NT=%ld)...\n", max_nb, max_nt);
    std::vector<float> all_xb(max_nb * d);
    generate_data(max_nb, d, all_xb);

    std::vector<float> all_xt(max_nt * d);
    generate_data(max_nt, d, all_xt);

    std::vector<idx_t> all_ids(max_nb);
    for (size_t i = 0; i < max_nb; ++i)
        all_ids[i] = (idx_t)i;

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp Memory

    GpuIndexIVFConfig config;
    config.device = 0;

    // Print Table Header
    printf("\n| %-10s | %-10s | %-15s | %-10s | %-15s | %-10s |\n",
           "NB",
           "nlist",
           "System",
           "Time(s)",
           "QPS (vec/s)",
           "Speedup");
    printf("|------------|------------|-----------------|------------|-----------------|------------|\n");

    // ==========================================
    // Execution Loop
    // ==========================================
    for (size_t nb : nb_list) {
        for (int nlist : nlist_list) {
            // Dynamically calculate training size (Faiss recommends ~39 * nlist)
            size_t nt = std::max((size_t)65536, (size_t)nlist * 40);
            if (nt > max_nt)
                nt = max_nt;

            double sivf_qps = 0;
            double vanilla_qps = 0;

            // --- Round 1: SIVF ---
            {
                // Pre define capacity with safety margin
                size_t max_vectors = nb * 2;
                // SIVF_SLAB_CAPACITY is 32. Add extra slabs for list heads.
                size_t slab_pool_size = max_vectors / 32 + (nlist * 2);

                GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
                index.initSlabManager(max_vectors, slab_pool_size);

                // Train
                index.train(nt, all_xt.data());

                // Benchmark Addition
                cudaDeviceSynchronize();
                double t0 = elapsed();
                
                // SIVF supports add_with_ids directly via inheritance
                index.add_with_ids(nb, all_xb.data(), all_ids.data());
                
                cudaDeviceSynchronize();
                double t1 = elapsed();

                double time_cost = t1 - t0;
                sivf_qps = nb / time_cost;

                printf("| %-10ld | %-10d | %-15s | %-10.4f | %-15.0f | %-10s |\n",
                       nb,
                       nlist,
                       "HPDIC SIVF",
                       time_cost,
                       sivf_qps,
                       "-");
            }

            // --- Round 2: Vanilla Faiss (IVFFlat) ---
            {
                GpuIndexIVFFlatConfig flatConfig;
                flatConfig.device = 0;
                faiss::gpu::GpuIndexIVFFlat index(
                        &res, d, nlist, METRIC_L2, flatConfig);

                // Train
                index.train(nt, all_xt.data());

                // Benchmark Addition
                cudaDeviceSynchronize();
                double t0 = elapsed();
                
                index.add_with_ids(nb, all_xb.data(), all_ids.data());
                
                cudaDeviceSynchronize();
                double t1 = elapsed();

                double time_cost = t1 - t0;
                vanilla_qps = nb / time_cost;

                printf("| %-10s | %-10s | %-15s | %-10.4f | %-15.0f | %-9.2fx |\n",
                       "\"",
                       "\"",
                       "Faiss IVFFlat",
                       time_cost,
                       vanilla_qps,
                       sivf_qps / vanilla_qps);
            }
            // Separator
            printf("|------------|------------|-----------------|------------|-----------------|------------|\n");
        }
    }

    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_add 
Preparing Data (Max NB=10000000, Max NT=262144)...

| NB         | nlist      | System          | Time(s)    | QPS (vec/s)     | Speedup    |
|------------|------------|-----------------|------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 11 (0.22 s, search 0.15 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 1000000    | 1024       | **SIVF**        | 0.1746     | 5728030         | -          |
| "          | "          | Vanilla         | 0.4812     | 2077930         | 2.76      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 7 (0.21 s, search 0.12 s): objective=594647 imbalance=2.248 nsplit=0         
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
| 1000000    | 2048       | **SIVF**        | 0.1884     | 5307344         | -          |
| "          | "          | Vanilla         | 0.5366     | 1863412         | 2.85      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 7 (0.60 s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2              
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 1000000    | 4096       | **SIVF**        | 0.2391     | 4182976         | -          |
| "          | "          | Vanilla         | 0.5995     | 1667996         | 2.51      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   127048 -> 629096
  > Data Buffer: 4000000 -> 20131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 11 (0.18 s, search 0.12 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 2000000    | 1024       | **SIVF**        | 0.3281     | 6095778         | -          |
| "          | "          | Vanilla         | 0.9456     | 2115055         | 2.88      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   129096 -> 629096
  > Data Buffer: 4000000 -> 20131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 7 (0.22 s, search 0.12 s): objective=594647 imbalance=2.248 nsplit=0         
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
| 2000000    | 2048       | **SIVF**        | 0.5595     | 3574601         | -          |
| "          | "          | Vanilla         | 1.1698     | 1709697         | 2.09      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   133192 -> 629096
  > Data Buffer: 4000000 -> 20131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.07 s
  Iteration 7 (0.61 s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2              
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 2000000    | 4096       | **SIVF**        | 0.4786     | 4178445         | -          |
| "          | "          | Vanilla         | 1.1477     | 1742637         | 2.40      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   252048 -> 1254096
  > Data Buffer: 8000000 -> 40131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 11 (0.19 s, search 0.13 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 4000000    | 1024       | **SIVF**        | 0.9982     | 4007265         | -          |
| "          | "          | Vanilla         | 1.7879     | 2237242         | 1.79      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   254096 -> 1254096
  > Data Buffer: 8000000 -> 40131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 7 (0.20 s, search 0.12 s): objective=594647 imbalance=2.248 nsplit=0         
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
| 4000000    | 2048       | **SIVF**        | 0.7557     | 5293120         | -          |
| "          | "          | Vanilla         | 1.9706     | 2029887         | 2.61      x |
|------------|------------|-----------------|------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   258192 -> 1254096
  > Data Buffer: 8000000 -> 40131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 7 (0.59 s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2              
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 4000000    | 4096       | **SIVF**        | 0.9498     | 4211634         | -          |
| "          | "          | Vanilla         | 2.3583     | 1696161         | 2.48      x |
|------------|------------|-----------------|------------|-----------------|------------|
cc@rtx6000:~/ElasticIVF/build$ 
 */