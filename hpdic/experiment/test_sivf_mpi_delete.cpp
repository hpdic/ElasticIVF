/**
 * test_sivf_mpi_delete.cpp
 *
 * Evaluation: Multi-GPU Scalability (Deletion)
 * Logic:
 * 1. SIVF: In-place GPU deletion.
 * 2. Vanilla: Try GPU delete -> Catch Exception -> Fallback to CPU Round-trip.
 */

#include <mpi.h>
#include <sys/time.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cstring> 
#include <numeric>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/GpuCloner.h> 
#include <faiss/impl/FaissException.h> // For try-catch
#include <faiss/impl/IDSelector.h>
#include <faiss/IndexIVFFlat.h>

using namespace faiss;
using namespace faiss::gpu;

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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int device_id = rank % num_gpus;
    cudaSetDevice(device_id);

    // Config
    int d = 128;
    int nlist = 4096; 
    size_t nb = 1000000;         // 1M vectors per GPU
    size_t n_delete = 10000;     // 10k deletions per GPU
    size_t max_nt = 256 * 1024;
    
    if (rank == 0) {
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "[MPI Deletion] Ranks: " << size << " | GPUs: " << num_gpus << std::endl;
        std::cout << "[Workload] Base: 1M vec/GPU | Delete: 10k vec/GPU" << std::endl;
        std::cout << "[Logic] Vanilla: Try GPU delete -> Fallback to CPU if failed" << std::endl;
    }

    // Data Gen
    std::vector<float> all_xb(nb * d);
    generate_data(nb, d, all_xb);

    std::vector<float> all_xt(max_nt * d);
    generate_data(max_nt, d, all_xt);

    std::vector<idx_t> all_ids(nb);
    idx_t id_offset = (idx_t)rank * nb;
    for (size_t i = 0; i < nb; ++i) all_ids[i] = id_offset + i;

    // Selector
    std::vector<idx_t> ids_to_delete(n_delete);
    for(size_t i = 0; i < n_delete; ++i) ids_to_delete[i] = id_offset + i;
    faiss::IDSelectorBatch selector(n_delete, ids_to_delete.data());

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 512); 

    GpuIndexIVFConfig config;
    config.device = device_id;

    if (rank == 0) {
        printf("\n| %-4s | %-10s | %-15s | %-15s | %-10s | %-10s |\n",
               "GPUs", "Del/GPU", "SIVF QPS", "Vanilla QPS", "Speedup", "Fallback?");
        printf("|------|------------|-----------------|-----------------|------------|------------|\n");
    }

    // ==========================================
    // Round 1: SIVF
    // ==========================================
    double sivf_total_qps = 0.0;
    {
        size_t max_vectors = nb * 2; 
        size_t slab_pool_size = max_vectors / 32 + (nlist * 2);

        GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
        index.initSlabManager(max_vectors, slab_pool_size);
        
        size_t nt = std::min(max_nt, (size_t)65536);
        index.train(nt, all_xt.data());
        index.add_with_ids(nb, all_xb.data(), all_ids.data());
        
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();
        index.remove_ids(selector); 
        cudaDeviceSynchronize();
        double t1 = MPI_Wtime();

        double local_qps = n_delete / (t1 - t0);
        MPI_Reduce(&local_qps, &sivf_total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // ==========================================
    // Round 2: Vanilla (Try GPU -> Catch -> CPU)
    // ==========================================
    double vanilla_total_qps = 0.0;
    int fallback_happened = 0; // 0=No, 1=Yes
    {
        GpuIndexIVFFlatConfig flatConfig;
        flatConfig.device = device_id;
        faiss::gpu::GpuIndexIVFFlat index(&res, d, nlist, METRIC_L2, flatConfig);
        
        size_t nt = std::min(max_nt, (size_t)65536);
        index.train(nt, all_xt.data());
        index.add_with_ids(nb, all_xb.data(), all_ids.data());

        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();
        
        try {
            // 1. Try Direct GPU Delete
            index.remove_ids(selector);
        } 
        catch (faiss::FaissException& e) {
            // 2. Catch "Not Implemented" and fallback to CPU Round-trip
            fallback_happened = 1;

            // Step A: Copy GPU -> CPU
            faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&index);
            
            // Step B: Delete on CPU
            cpu_index->remove_ids(selector);
            
            // Step C: Copy CPU -> GPU (Simulate restoring the service)
            // We create a new GPU index to measure the upload cost
            faiss::gpu::GpuIndexIVFFlat* new_gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(
                faiss::gpu::index_cpu_to_gpu(&res, device_id, cpu_index));
            
            delete cpu_index;
            delete new_gpu_index;
        }

        cudaDeviceSynchronize();
        double t1 = MPI_Wtime();

        double local_qps = n_delete / (t1 - t0);
        MPI_Reduce(&local_qps, &vanilla_total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    // Check if ALL ranks triggered fallback
    int global_fallback = 0;
    MPI_Reduce(&fallback_happened, &global_fallback, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // ==========================================
    // Report
    // ==========================================
    if (rank == 0) {
        printf("| %-4d | %-10ld | %-15.0f | %-15.0f | %-10.1fx | %-10s |\n",
               size, n_delete, sivf_total_qps, vanilla_total_qps, 
               sivf_total_qps / vanilla_total_qps,
               (global_fallback ? "YES" : "NO"));
    }

    MPI_Finalize();
    return 0;
}

/** Example output:
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root -np 1 ./faiss/gpu/test_sivf_mpi_delete

==========================================================
[MPI Deletion] Ranks: 1 | GPUs: 4
[Workload] Base: 1M vec/GPU | Delete: 10k vec/GPU
[Logic] Vanilla: Try GPU delete -> Fallback to CPU if failed

| GPUs | Del/GPU    | SIVF QPS        | Vanilla QPS     | Speedup    | Fallback?  |
|------|------------|-----------------|-----------------|------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 6 (0.36 s, search 0.11 s): objective=334205 imbalance=1.717 nsplit=2         
  Converged at iteration 6: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| 1    | 10000      | 13908399        | 6297            | 2208.6    x | YES        |
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ 




(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ mpirun --allow-run-as-root -np 2 ./faiss/gpu/test_si
vf_mpi_delete

==========================================================
[MPI Deletion] Ranks: 2 | GPUs: 4
[Workload] Base: 1M vec/GPU | Delete: 10k vec/GPU
[Logic] Vanilla: Try GPU delete -> Fallback to CPU if failed

| GPUs | Del/GPU    | SIVF QPS        | Vanilla QPS     | Speedup    | Fallback?  |
|------|------------|-----------------|-----------------|------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[HPDIC MOD] Faiss GPU initialized on device ID: 1

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   70692 -> 316596
  > Data Buffer: 2000000 -> 10131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
Clustering 65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
  Iteration 6 (0.36 s, search 0.10 s): objective=334205 imbalance=1.717 nsplit=2         
  Converged at iteration 6: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
  Iteration 6 (0.36 s, search 0.11 s): objective=334205 imbalance=1.717 nsplit=2       
  Converged at iteration 6: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| 2    | 10000      | 34050875        | 12026           | 2831.6    x | YES        |
(myenv) donzhao@node0:~/hpdic/ElasticIVF/build$ 





*/