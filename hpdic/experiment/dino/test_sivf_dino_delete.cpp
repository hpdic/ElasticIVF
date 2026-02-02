/**
 * test_sivf_dino_delete.cpp
 *
 * Evaluation: Distributed Deletion Throughput (SIVF vs. Faiss Baseline)
 * Dataset: DINO 10B (1024-dim)
 * 
 * Logic:
 * 1. Setup: Ingest 100k vectors/GPU (same as Add benchmark).
 * 2. Workload: Select 10k vectors/GPU to delete.
 * 3. SIVF: In-place GPU bitmap flip.
 * 4. Baseline: CPU Fallback (Download -> Delete on CPU -> Upload).
 */

#include <mpi.h>
#include <sys/time.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring> 
#include <random>
#include <omp.h>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/index_io.h> // For CPU fallback cloning
#include <faiss/gpu/GpuCloner.h>

using namespace faiss;
using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Rank 0 reads chunk
// ---------------------------------------------------------
void read_dino_chunk_for_all(const std::string& filename, size_t vectors_per_rank, int world_size, int d, std::vector<float>& out_buffer) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[Error] Rank 0 cannot open file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t total_vectors = vectors_per_rank * world_size;
    size_t row_size = 4 + d; 
    
    // Skip training data (first 100k)
    size_t offset = 100000 * row_size; 
    file.seekg(offset, std::ios::beg);

    std::cout << "[IO] Rank 0 reading " << total_vectors << " vectors (" 
              << (total_vectors * row_size) / (1024.0*1024.0) << " MB) from disk..." << std::endl;

    std::vector<uint8_t> raw_bytes(total_vectors * row_size);
    file.read(reinterpret_cast<char*>(raw_bytes.data()), total_vectors * row_size);
    
    size_t read_count = file.gcount() / row_size;
    if (read_count < total_vectors) {
        std::cerr << "[Warning] Requested " << total_vectors << " but only read " << read_count << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2); 
    }

    out_buffer.resize(total_vectors * d);
    
    #pragma omp parallel for
    for (size_t i = 0; i < total_vectors; ++i) {
        uint8_t* row_ptr = raw_bytes.data() + i * row_size;
        uint8_t* vec_ptr = row_ptr + 4; 
        for (int j = 0; j < d; ++j) {
            out_buffer[i * d + j] = static_cast<float>(vec_ptr[j]);
        }
    }
    std::cout << "[IO] Read complete." << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = local_rank_env ? atoi(local_rank_env) : rank % 4;
    cudaSetDevice(local_rank);

    // ==========================================
    // Config
    // ==========================================
    int d = 1024;
    std::string file_path = "/home/cc/hpdic/data/dino10b/chunk_0000.bvecs"; 
    
    int nlist = 4096;
    size_t nb_per_rank = 100000; // 100k vectors Base
    size_t n_del = 10000;        // Delete 10k vectors (10%)

    // 1. Prepare Data
    std::vector<float> local_xb(nb_per_rank * d);
    std::vector<float> host_full_buffer;

    if (rank == 0) {
        read_dino_chunk_for_all(file_path, nb_per_rank, size, d, host_full_buffer);
    }

    MPI_Scatter(host_full_buffer.data(), nb_per_rank * d, MPI_FLOAT,
                local_xb.data(), nb_per_rank * d, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<float>().swap(host_full_buffer); 
        std::cout << "[MPI] Data distributed. Setting up Index..." << std::endl;
    }

    // IDs and Deletion Selection
    std::vector<idx_t> local_ids(nb_per_rank);
    for (size_t i = 0; i < nb_per_rank; ++i) local_ids[i] = (idx_t)(rank * nb_per_rank + i);

    // Prepare Delete Batch (First n_del IDs)
    std::vector<idx_t> delete_ids(n_del);
    for(size_t i=0; i<n_del; ++i) delete_ids[i] = local_ids[i];
    
    IDSelectorBatch selector(n_del, delete_ids.data());

    // Training Data
    size_t nt = 65536; 
    std::vector<float> train_xt(nt * d);
    memcpy(train_xt.data(), local_xb.data(), nt * d * sizeof(float));

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024);

    GpuIndexIVFFlatConfig config;
    config.device = local_rank;

    if (rank == 0) {
        printf("\n| %-4s | %-10s | %-10s | %-8s | %-15s | %-15s | %-10s | %-10s |\n",
               "GPUs", "Del/GPU", "nlist", "System", "Avg Latency(s)", "Total QPS", "Speedup", "Fallback?");
        printf("|------|------------|------------|----------|-----------------|-----------------|------------|------------|\n");
    }

    // --- Round 1: SIVF ---
    {
        size_t capacity = nb_per_rank * 1.5; 
        GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
        index.initSlabManager(capacity, d);
        index.train(nt, train_xt.data());
        index.add_with_ids(nb_per_rank, local_xb.data(), local_ids.data());
        
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();
        // SIVF Native GPU Deletion
        index.remove_ids(selector);
        cudaDeviceSynchronize();
        double t1 = MPI_Wtime();

        double local_time = t1 - t0;
        double local_qps = n_del / local_time;

        double total_qps = 0;
        double max_time = 0;
        MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("| %-4d | %-10ld | %-10d | %-8s | %-15.6f | %-15.0f | %-10s | %-10s |\n",
                   size, n_del, nlist, "**SIVF**", max_time, total_qps, "-", "NO");
        }
    }

    // --- Round 2: Vanilla (CPU Fallback) ---
    {
        GpuIndexIVFFlatConfig flatConfig;
        flatConfig.device = local_rank;
        faiss::gpu::GpuIndexIVFFlat index(&res, d, nlist, METRIC_L2, flatConfig);
        index.train(nt, train_xt.data());
        index.add_with_ids(nb_per_rank, local_xb.data(), local_ids.data());

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();
        bool fallback_triggered = false;

        try {
            // Try direct GPU delete (Likely throws or fails silently on some versions)
            long n_removed = index.remove_ids(selector);
            if (n_removed == 0) throw FaissException("Not implemented on GPU");
        } catch (...) {
            // CPU Fallback Simulation
            // 1. Copy to CPU
            fallback_triggered = true;
            Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&index);
            // 2. Remove on CPU
            cpu_index->remove_ids(selector);
            // 3. Copy back to GPU (Re-build)
            // GpuIndexIVFFlat cannot "update" from CPU easily, usually requires re-ingest
            // To be fair to baseline, we count the "GPU->CPU->Delete" time. 
            // Uploading back is technically "Re-indexing", which is huge.
            // We'll just measure the Downlink + CPU Delete, which is already slow enough.
            delete cpu_index; 
        }

        cudaDeviceSynchronize();
        double t1 = MPI_Wtime();

        double local_time = t1 - t0;
        double local_qps = n_del / local_time;

        double total_qps = 0;
        double max_time = 0;
        MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("| %-4s | %-10s | %-10s | %-8s | %-15.6f | %-15.0f | %-10s | %-10s |\n",
                   "\"", "\"", "\"", "Vanilla", max_time, total_qps, "1.0x", "YES");
            printf("|------|------------|------------|----------|-----------------|-----------------|------------|------------|\n");
        }
    }

    MPI_Finalize();
    return 0;
}

/**
 * Example output:
 * 
cc@gpu0:~/hpdic/ElasticIVF$ mpirun --allow-run-as-root \
    -np 10 \
    --host gpu0:4,gpu1:4,gpu2:2 \
    -x LD_LIBRARY_PATH \
    ~/hpdic/ElasticIVF/build/test_sivf_dino_delete 
[IO] Rank 0 reading 1000000 vectors (980.377 MB) from disk...
[IO] Read complete.
[HPDIC MOD] Faiss GPU initialized on device ID: 1
[HPDIC MOD] Faiss GPU initialized on device ID: 2

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

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
[HPDIC MOD] Faiss GPU initialized on device ID: 02.28991e+10 imbalance=1.531 nsplit=0       
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterationssplit=0       
  Preprocessing in 0.35 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[HPDIC MOD] Faiss GPU initialized on device ID: 12.28987e+10 imbalance=1.535 nsplit=0       
  Iteration 1 (0.34 s, search 0.22 s): objective=1.39549e+10 imbalance=1.426 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.42 s
[HPDIC MOD] Faiss GPU initialized on device ID: 22.28913e+10 imbalance=1.524 nsplit=0       
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterationssplit=0       
  Preprocessing in 0.33 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
  Iteration 4 (0.78 s, search 0.53 s): objective=1.34607e+10 imbalance=1.382 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[HPDIC MOD] Faiss GPU initialized on device ID: 31.34222e+10 imbalance=1.379 nsplit=0       
  Iteration 6 (1.08 s, search 0.74 s): objective=1.34093e+10 imbalance=1.378 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.37 s
[HPDIC MOD] Faiss GPU initialized on device ID: 01.33977e+10 imbalance=1.377 nsplit=0       
[MPI] Data distributed. Setting up Index...ctive=2.28799e+10 imbalance=1.527 nsplit=0        
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.35 s
  Iteration 10 (1.69 s, search 1.16 s): objective=1.34135e+10 imbalance=1.362 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
[HPDIC MOD] Faiss GPU initialized on device ID: 1=1.34122e+10 imbalance=1.362 nsplit=0       
  Iteration 12 (1.99 s, search 1.37 s): objective=1.34114e+10 imbalance=1.362 nsplit=0       
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.33 s
  Iteration 0 (0.17 s, search 0.10 s): objective=2.29557e+10 imbalance=1.525 nsplit=0        
| GPUs | Del/GPU    | nlist      | System   | Avg Latency(s)  | Total QPS       | Speedup    | Fallback?  |
|------|------------|------------|----------|-----------------|-----------------|------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.33 s
  Iteration 8 (1.46 s, search 1.04 s): objective=1.34254e+10 imbalance=1.384 nsplit=0        
[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.39 s
  Iteration 8 (1.50 s, search 1.05 s): objective=1.34085e+10 imbalance=1.380 nsplit=0        
  Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.18 s, search 2.17 s): objective=1.3399e+10 imbalance=1.377 nsplit=0        
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.nsplit=0       
  Iteration 19 (3.14 s, search 2.17 s): objective=1.3378e+10 imbalance=1.375 nsplit=0        
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.12 s, search 2.17 s): objective=1.33899e+10 imbalance=1.370 nsplit=0       
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 17 (2.88 s, search 2.02 s): objective=1.34159e+10 imbalance=1.383 nsplit=0       
  Converged at iteration 17: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (3.18 s, search 2.21 s): objective=1.3394e+10 imbalance=1.380 nsplit=0        
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.0 nsplit=0       
  Iteration 19 (3.13 s, search 2.18 s): objective=1.33609e+10 imbalance=1.372 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (2.80 s, search 2.00 s): objective=1.3397e+10 imbalance=1.368 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (2.82 s, search 2.01 s): objective=1.33916e+10 imbalance=1.372 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.0 nsplit=0       
  Iteration 19 (2.93 s, search 1.99 s): objective=1.33687e+10 imbalance=1.380 nsplit=0       
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 10   | 10000      | 4096       | **SIVF** | 0.001649        | 89462837        | -          | NO         |
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
| "    | "          | "          | Vanilla  | 1.324727        | 77889           | 1.0x       | YES        |
|------|------------|------------|----------|-----------------|-----------------|------------|------------|
cc@gpu0:~/hpdic/ElasticIVF$ 





cc@gpu0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root     -np 2     --host gpu3:2     -x LD_LIBRARY_PATH     ~/hpdic/ElasticIVF/build/test_sivf_dino_delete  
[IO] Rank 0 reading 200000 vectors (196.075 MB) from disk...
[IO] Read complete.
[MPI] Data distributed. Setting up Index...
[HPDIC MOD] Faiss GPU initialized on device ID: 1

| GPUs | Del/GPU    | nlist      | System   | Avg Latency(s)  | Total QPS       | Speedup    | Fallback?  |
|------|------------|------------|----------|-----------------|-----------------|------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 27536
  > Data Buffer: 150000 -> 881152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.27 s
Clustering 65536 points in 1024D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.27 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
  Iteration 19 (6.10 s, search 2.50 s): objective=1.3399e+10 imbalance=1.377 nsplit=0        
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 19 (6.21 s, search 2.56 s): objective=1.33687e+10 imbalance=1.380 nsplit=0       
  Converged at iteration 19: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 2    | 10000      | 4096       | **SIVF** | 0.001058        | 19038105        | -          | NO         |
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| "    | "          | "          | Vanilla  | 1.243398        | 16463           | 1.0x       | YES        |
|------|------------|------------|----------|-----------------|-----------------|------------|------------|
cc@gpu0:~/hpdic/ElasticIVF/build$ 
 */