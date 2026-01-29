/**
 * test_sivf_mpi_insert.cpp
 *
 * Evaluation Section: Multi-GPU Scalability (Insertion)
 * Logic: Weak Scaling. Each GPU handles 'nb' vectors.
 * Total System Throughput = Sum(QPS of all GPUs).
 */

#include <mpi.h>
#include <sys/time.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstring> 

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss;
using namespace faiss::gpu;

// Fast random data generation (Same as original)
void generate_data(size_t n, int d, std::vector<float>& data) {
    size_t chunk = std::min(n, (size_t)10000);
    for (size_t i = 0; i < chunk * d; ++i) {
        data[i] = (float)drand48();
    }
    for (size_t i = chunk; i < n; ++i) {
        memcpy(data.data() + i * d,
               data.data() + (i % chunk) * d,
               d * sizeof(float));
    }
}

int main(int argc, char** argv) {
    // 1. MPI Init
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. Bind to GPU
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int device_id = rank % num_gpus;
    cudaSetDevice(device_id);

    // ==========================================
    // Experiment Configuration
    // ==========================================
    int d = 128;
    std::vector<int> nlist_list = {1024, 2048, 4096};
    std::vector<size_t> nb_list = {1000000}; // Per-GPU Load

    // Prepare Max Data Buffers (Reuse to save allocation time)
    size_t max_nb = 10000000; 
    size_t max_nt = 256 * 1024;
    
    if (rank == 0) {
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "[MPI Scaling] Ranks: " << size << " | GPUs: " << num_gpus << std::endl;
        std::cout << "[Setup] Generating Synthetic Data (" << max_nb << " vectors)..." << std::endl;
    }

    std::vector<float> all_xb(max_nb * d);
    generate_data(max_nb, d, all_xb);

    std::vector<float> all_xt(max_nt * d);
    generate_data(max_nt, d, all_xt);

    std::vector<idx_t> all_ids(max_nb);
    for (size_t i = 0; i < max_nb; ++i) all_ids[i] = (idx_t)(rank * max_nb + i); // Unique IDs across ranks

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp

    // Common Config
    GpuIndexIVFConfig config;
    config.device = device_id;

    if (rank == 0) {
        printf("\n| %-4s | %-10s | %-10s | %-8s | %-15s | %-15s | %-10s |\n",
               "GPUs", "NB/GPU", "nlist", "System", "Avg Latency(s)", "Total QPS", "Speedup");
        printf("|------|------------|------------|----------|-----------------|-----------------|------------|\n");
    }

    // ==========================================
    // Execution Loop
    // ==========================================
    for (size_t nb : nb_list) {
        for (int nlist : nlist_list) {
            size_t nt = std::max((size_t)65536, (size_t)nlist * 40);
            if (nt > max_nt) nt = max_nt;

            // --- Round 1: SIVF ---
            {
                size_t max_vectors = nb * 2; 
                size_t slab_pool_size = max_vectors / 32 + (nlist * 2);

                GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
                index.initSlabManager(max_vectors, slab_pool_size);
                index.train(nt, all_xt.data());
                
                // Sync before timing
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                double t0 = MPI_Wtime();
                index.add_with_ids(nb, all_xb.data(), all_ids.data());
                cudaDeviceSynchronize();
                double t1 = MPI_Wtime();

                double local_time = t1 - t0;
                double local_qps = nb / local_time;

                // Aggregate
                double total_sivf_qps = 0;
                double max_time = 0;
                MPI_Reduce(&local_qps, &total_sivf_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank == 0) {
                    printf("| %-4d | %-10ld | %-10d | %-8s | %-15.4f | %-15.0f | %-10s |\n",
                           size, nb, nlist, "**SIVF**", max_time, total_sivf_qps, "-");
                }
            }

            // --- Round 2: Vanilla Faiss (Optional comparison) ---
            {
                GpuIndexIVFFlatConfig flatConfig;
                flatConfig.device = device_id;
                faiss::gpu::GpuIndexIVFFlat index(&res, d, nlist, METRIC_L2, flatConfig);
                index.train(nt, all_xt.data());

                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                double t0 = MPI_Wtime();
                index.add_with_ids(nb, all_xb.data(), all_ids.data());
                cudaDeviceSynchronize();
                double t1 = MPI_Wtime();

                double local_time = t1 - t0;
                double local_qps = nb / local_time;

                double total_vanilla_qps = 0;
                double max_time = 0;
                MPI_Reduce(&local_qps, &total_vanilla_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank == 0) {
                    // Calculate relative speedup (SIVF vs Vanilla)
                    // Note: We need the PREVIOUS SIVF QPS to calculate speedup, 
                    // but for simplicity here we just print the raw numbers.
                    printf("| %-4s | %-10s | %-10s | %-8s | %-15.4f | %-15.0f | %-10s |\n",
                           "\"", "\"", "\"", "Vanilla", max_time, total_vanilla_qps, "1.0x");
                     printf("|------|------------|------------|----------|-----------------|-----------------|------------|\n");
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}

/** Example output:
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root -np 1 ./faiss/gpu/test_sivf_mpi_insert 

==========================================================
[MPI Scaling] Ranks: 1 | GPUs: 4
[Setup] Generating Synthetic Data (10000000 vectors)...

| GPUs | NB/GPU     | nlist      | System   | Avg Latency(s)  | Total QPS       | Speedup    |
|------|------------|------------|----------|-----------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 11 (0.35 s, search 0.10 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 1    | 1000000    | 1024       | **SIVF** | 0.2260          | 4425399         | -          |
| "    | "          | "          | Vanilla  | 0.5288          | 1890938         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 7 (0.31 s, search 0.09 s): objective=594647 imbalance=2.248 nsplit=0         
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
| 1    | 1000000    | 2048       | **SIVF** | 0.1466          | 6820133         | -          |
| "    | "          | "          | Vanilla  | 0.5571          | 1794871         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.08 s
  Iteration 7 (0.78 s, search 0.24 s): objective=842149 imbalance=1.872 nsplit=2              
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 1    | 1000000    | 4096       | **SIVF** | 0.1930          | 5181931         | -          |
| "    | "          | "          | Vanilla  | 0.6197          | 1613728         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ 





(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root -np 2 ./faiss/gpu/test_si
vf_mpi_insert 

==========================================================
[MPI Scaling] Ranks: 2 | GPUs: 4
[Setup] Generating Synthetic Data (10000000 vectors)...

| GPUs | NB/GPU     | nlist      | System   | Avg Latency(s)  | Total QPS       | Speedup    |
|------|------------|------------|----------|-----------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
[HPDIC MOD] Faiss GPU initialized on device ID: 1
  Iteration 4 (0.16 s, search 0.05 s): objective=553472 imbalance=1.925 nsplit=0        
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterationst=0       
  Preprocessing in 0.03 s
  Iteration 11 (0.36 s, search 0.11 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
  Iteration 11 (0.36 s, search 0.11 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 2    | 1000000    | 1024       | **SIVF** | 0.2607          | 11027819        | -          |
| "    | "          | "          | Vanilla  | 0.5186          | 3894296         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 7 (0.33 s, search 0.11 s): objective=594647 imbalance=2.248 nsplit=0         
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
  Iteration 7 (0.33 s, search 0.11 s): objective=594647 imbalance=2.248 nsplit=0       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
| 2    | 1000000    | 2048       | **SIVF** | 0.1590          | 12665461        | -          |
| "    | "          | "          | Vanilla  | 0.5825          | 3469373         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)


[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.07 s
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.09 s
  Iteration 7 (0.78 s, search 0.24 s): objective=842149 imbalance=1.872 nsplit=2              
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 7 (0.78 s, search 0.25 s): objective=842149 imbalance=1.872 nsplit=2       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 2    | 1000000    | 4096       | **SIVF** | 0.3307          | 7913263         | -          |
| "    | "          | "          | Vanilla  | 0.6473          | 3121210         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ 






(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root -np 4 ./faiss/gpu/test_sivf_mpi_insert

==========================================================
[MPI Scaling] Ranks: 4 | GPUs: 4
[Setup] Generating Synthetic Data (10000000 vectors)...

| GPUs | NB/GPU     | nlist      | System   | Avg Latency(s)  | Total QPS       | Speedup    |
|------|------------|------------|----------|-----------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
[HPDIC MOD] Faiss GPU initialized on device ID: 3553151 imbalance=1.923 nsplit=0        
[HPDIC MOD] Faiss GPU initialized on device ID: 1
[HPDIC MOD] Faiss GPU initialized on device ID: 2553140 imbalance=1.923 nsplit=0       
  Iteration 11 (0.25 s, search 0.11 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)


[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   64548 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 11 (0.26 s, search 0.12 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
  Iteration 11 (0.32 s, search 0.13 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
  Iteration 11 (0.32 s, search 0.13 s): objective=553137 imbalance=1.923 nsplit=0       
  Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 4    | 1000000    | 1024       | **SIVF** | 0.2293          | 17749497        | -          |
| "    | "          | "          | Vanilla  | 0.5890          | 6862959         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)


[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)


[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66596 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
Clustering 81920 points in 128D to 2048 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 7 (0.26 s, search 0.11 s): objective=594647 imbalance=2.248 nsplit=0         
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
  Iteration 7 (0.25 s, search 0.12 s): objective=594647 imbalance=2.248 nsplit=0       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
  Iteration 7 (0.28 s, search 0.13 s): objective=594647 imbalance=2.248 nsplit=0       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
  Iteration 7 (0.28 s, search 0.13 s): objective=594647 imbalance=2.248 nsplit=0       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 2048 centroids.
| 4    | 1000000    | 2048       | **SIVF** | 0.2563          | 15781606        | -          |
| "    | "          | "          | Vanilla  | 0.6497          | 6316239         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)


[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)


[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.10 s
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.07 s
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.09 s
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.10 s
  Iteration 7 (0.64 s, search 0.29 s): objective=842149 imbalance=1.872 nsplit=2              
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 7 (0.66 s, search 0.29 s): objective=842149 imbalance=1.872 nsplit=2       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 7 (0.66 s, search 0.29 s): objective=842149 imbalance=1.872 nsplit=2       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 7 (0.66 s, search 0.29 s): objective=842149 imbalance=1.872 nsplit=2       
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 4    | 1000000    | 4096       | **SIVF** | 0.3442          | 12675814        | -          |
| "    | "          | "          | Vanilla  | 0.7093          | 5697525         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ 
*/