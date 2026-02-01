/**
 * test_sivf_dino_add.cpp
 *
 * Evaluation: Distributed Ingestion (Rank 0 Reads -> MPI Scatter -> SIVF Insert)
 * Dataset: DINO 10B (1024-dim)
 * 
 * Logic:
 * 1. Rank 0 loads ALL data from disk (IO Bottleneck handled by Master).
 * 2. Rank 0 scatters data to all workers.
 * 3. Parallel Insertion & Benchmarking.
 */

#include <mpi.h>
#include <sys/time.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring> 
#include <omp.h>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss;
using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Rank 0 reads a large chunk for everyone
// ---------------------------------------------------------
void read_dino_chunk_for_all(const std::string& filename, size_t vectors_per_rank, int world_size, int d, std::vector<float>& out_buffer) {
    // Only Rank 0 calls this effectively
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[Error] Rank 0 cannot open file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t total_vectors = vectors_per_rank * world_size;
    size_t row_size = 4 + d; // 4 bytes header + 1024 bytes data
    
    // Skip training data (first 100k) to be safe
    size_t offset = 100000 * row_size; 
    file.seekg(offset, std::ios::beg);

    std::cout << "[IO] Rank 0 reading " << total_vectors << " vectors (" 
              << (total_vectors * row_size) / (1024.0*1024.0) << " MB) from disk..." << std::endl;

    std::vector<uint8_t> raw_bytes(total_vectors * row_size);
    file.read(reinterpret_cast<char*>(raw_bytes.data()), total_vectors * row_size);
    
    size_t read_count = file.gcount() / row_size;
    if (read_count < total_vectors) {
        std::cerr << "[Warning] Requested " << total_vectors << " but only read " << read_count << std::endl;
        // Pad with zeros if necessary or abort
        MPI_Abort(MPI_COMM_WORLD, 2); 
    }

    // Convert to float
    out_buffer.resize(total_vectors * d);
    
    #pragma omp parallel for
    for (size_t i = 0; i < total_vectors; ++i) {
        uint8_t* row_ptr = raw_bytes.data() + i * row_size;
        uint8_t* vec_ptr = row_ptr + 4; // Skip header
        for (int j = 0; j < d; ++j) {
            out_buffer[i * d + j] = static_cast<float>(vec_ptr[j]);
        }
    }
    std::cout << "[IO] Read complete. Preparing to Scatter..." << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Bind GPU
    const char* local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = local_rank_env ? atoi(local_rank_env) : rank % 4;
    cudaSetDevice(local_rank);

    // Config
    int d = 1024;
    // Please verify this path exists on gpu0!
    // NO TILDE (~), use Absolute Path
    std::string file_path = "/data/dino10b/chunk_0000.bvecs"; 
    
    std::vector<int> nlist_list = {4096};
    size_t nb_per_rank = 100000; // 100k vectors per GPU (Total 1M vectors for 10 GPUs)
    // 100k * 1024 floats = ~400MB RAM per GPU. Safe.

    // 1. Prepare Data (Rank 0 Reads, Everyone Receives)
    std::vector<float> local_xb(nb_per_rank * d); // Buffer for this GPU
    std::vector<float> host_full_buffer;          // Buffer for Rank 0 to read disk

    if (rank == 0) {
        // Rank 0 reads (100k * 10) = 1M vectors -> ~4GB RAM (Host RAM is 125GB, Safe)
        read_dino_chunk_for_all(file_path, nb_per_rank, size, d, host_full_buffer);
    }

    // Scatter Data (Rank 0 -> All Ranks)
    // Note: MPI_Scatter sends contiguous chunks. 
    // Rank 0 gets chunk 0, Rank 1 gets chunk 1... exactly what we want.
    MPI_Scatter(host_full_buffer.data(), nb_per_rank * d, MPI_FLOAT,
                local_xb.data(), nb_per_rank * d, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Free Host RAM on Rank 0 to save memory for subsequent steps if needed
    if (rank == 0) {
        std::vector<float>().swap(host_full_buffer); 
        std::cout << "[MPI] Data distributed. Starting Benchmarks..." << std::endl;
    }

    // Prepare IDs
    std::vector<idx_t> local_ids(nb_per_rank);
    for (size_t i = 0; i < nb_per_rank; ++i) local_ids[i] = (idx_t)(rank * nb_per_rank + i);

    // Generate/Load Training Data (Reuse first part of local data for simplicity)
    size_t nt = 65536; 
    std::vector<float> train_xt(nt * d);
    // Everyone just uses the first 65k of their own data to train 
    // (In strict theory we should broadcast a common train set, but for throughput benchmark this is fine)
    memcpy(train_xt.data(), local_xb.data(), nt * d * sizeof(float));

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp

    GpuIndexIVFFlatConfig config;
    config.device = local_rank;

    if (rank == 0) {
        printf("\n| %-4s | %-10s | %-10s | %-8s | %-15s | %-15s | %-10s |\n",
               "GPUs", "NB/GPU", "nlist", "System", "Avg Latency(s)", "Total QPS", "Speedup");
        printf("|------|------------|------------|----------|-----------------|-----------------|------------|\n");
    }

    // Benchmarking Loop
    for (int nlist : nlist_list) {
        
        // --- Round 1: SIVF ---
        {
            size_t capacity = nb_per_rank * 1.5; 
            GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
            index.initSlabManager(capacity, d);
            index.train(nt, train_xt.data());
            
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            double t0 = MPI_Wtime();
            index.add_with_ids(nb_per_rank, local_xb.data(), local_ids.data());
            cudaDeviceSynchronize();
            double t1 = MPI_Wtime();

            double local_time = t1 - t0;
            double local_qps = nb_per_rank / local_time;

            double total_qps = 0;
            double max_time = 0;
            MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("| %-4d | %-10ld | %-10d | %-8s | %-15.4f | %-15.0f | %-10s |\n",
                       size, nb_per_rank, nlist, "**SIVF**", max_time, total_qps, "-");
            }
        }

        // --- Round 2: Vanilla ---
        {
            GpuIndexIVFFlatConfig flatConfig;
            flatConfig.device = local_rank;
            faiss::gpu::GpuIndexIVFFlat index(&res, d, nlist, METRIC_L2, flatConfig);
            index.train(nt, train_xt.data());

            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);

            double t0 = MPI_Wtime();
            index.add_with_ids(nb_per_rank, local_xb.data(), local_ids.data());
            cudaDeviceSynchronize();
            double t1 = MPI_Wtime();

            double local_time = t1 - t0;
            double local_qps = nb_per_rank / local_time;

            double total_qps = 0;
            double max_time = 0;
            MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("| %-4s | %-10s | %-10s | %-8s | %-15.4f | %-15.0f | %-10s |\n",
                       "\"", "\"", "\"", "Vanilla", max_time, total_qps, "1.0x");
                printf("|------|------------|------------|----------|-----------------|-----------------|------------|\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}

/** Example output:
 * 
cc@gpu0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root     -np 10     --host gpu0:4,gpu1:4,gpu2:2     -x LD_LIBRARY_PATH     ~/hpdic/ElasticIVF/build/test_sivf_dino_add 
[IO] Rank 0 reading 1000000 vectors (980.377 MB) from disk...
[IO] Read complete. Preparing to Scatter...
[HPDIC MOD] Faiss GPU initialized on device ID: 1

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[HPDIC MOD] Faiss GPU initialized on device ID: 2

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[HPDIC MOD] Faiss GPU initialized on device ID: 3
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.33 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.33 s
[HPDIC MOD] Faiss GPU initialized on device ID: 0
  Iteration 2 (0.46 s, search 0.30 s): objective=1.36734e+10 imbalance=1.398 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.37 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
[HPDIC MOD] Faiss GPU initialized on device ID: 12.28987e+10 imbalance=1.535 nsplit=0       
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.33 s
  Iteration 3 (0.64 s, search 0.43 s): objective=1.35494e+10 imbalance=1.376 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[HPDIC MOD] Faiss GPU initialized on device ID: 21.34904e+10 imbalance=1.371 nsplit=0       
  Iteration 5 (0.94 s, search 0.64 s): objective=1.34581e+10 imbalance=1.369 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...9 nsplit=0       
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
  Preprocessing in 0.35 s
[HPDIC MOD] Faiss GPU initialized on device ID: 31.34394e+10 imbalance=1.366 nsplit=0       
  Iteration 9 (1.52 s, search 1.03 s): objective=1.34099e+10 imbalance=1.379 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterationssplit=0       
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
  Preprocessing in 0.37 s
[HPDIC MOD] Faiss GPU initialized on device ID: 01.33977e+10 imbalance=1.377 nsplit=0        
  Iteration 3 (0.63 s, search 0.43 s): objective=1.35588e+10 imbalance=1.397 nsplit=0        
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[MPI] Data distributed. Starting Benchmarks...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.37 s
[HPDIC MOD] Faiss GPU initialized on device ID: 11.33857e+10 imbalance=1.376 nsplit=0        
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.33 s
  Iteration 3 (0.64 s, search 0.44 s): objective=1.35416e+10 imbalance=1.391 nsplit=0        
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
  Iteration 11 (1.86 s, search 1.30 s): objective=1.33814e+10 imbalance=1.376 nsplit=0       
| GPUs | NB/GPU     | nlist      | System   | Avg Latency(s)  | Total QPS       | Speedup    |
|------|------------|------------|----------|-----------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0
  Iteration 14 (2.32 s, search 1.58 s): objective=1.34108e+10 imbalance=1.362 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...3 nsplit=0       
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.37 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.46 s
  Iteration 19 (3.06 s, search 2.07 s): objective=1.3399e+10 imbalance=1.377 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 18 (2.94 s, search 2.01 s): objective=1.34106e+10 imbalance=1.362 nsplit=0       
  Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.07 s, search 2.14 s): objective=1.3378e+10 imbalance=1.375 nsplit=0        
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.06 s, search 2.14 s): objective=1.33899e+10 imbalance=1.370 nsplit=0       
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 17 (2.82 s, search 1.96 s): objective=1.34159e+10 imbalance=1.383 nsplit=0       
  Converged at iteration 17: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.07 s, search 2.14 s): objective=1.3394e+10 imbalance=1.380 nsplit=0        
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (2.74 s, search 1.97 s): objective=1.3397e+10 imbalance=1.368 nsplit=0        
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.11 s, search 2.16 s): objective=1.33609e+10 imbalance=1.372 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (2.83 s, search 2.07 s): objective=1.33916e+10 imbalance=1.372 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (2.93 s, search 1.96 s): objective=1.33687e+10 imbalance=1.380 nsplit=0       
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 10   | 100000     | 4096       | **SIVF** | 0.3250          | 3402566         | -          |
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| "    | "          | "          | Vanilla  | 0.7143          | 1496738         | 1.0x       |
|------|------------|------------|----------|-----------------|-----------------|------------|
cc@gpu0:~/hpdic/ElasticIVF/build$ 
 */