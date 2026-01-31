/**
 * test_sivf_dino_add.cpp
 *
 * Evaluation: Distributed Ingestion Throughput (SIVF vs. Faiss Baseline)
 * Dataset: DINO 10B (1024-dim), Streamed from Disk
 * 
 * Logic:
 * 1. Each MPI Rank reads a unique partition of the DINO dataset from disk.
 * 2. Round 1: Build SIVF Index.
 * 3. Round 2: Build Faiss GPU IVF (Baseline).
 * 4. Report QPS and Speedup.
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
// Helper: Read DINO .bvecs partition
// Format: 4 bytes (int dim) + d bytes (uint8 data)
// ---------------------------------------------------------
void read_dino_part(const std::string& filename, int rank, size_t n, int d, std::vector<float>& out_data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        if (rank == 0) std::cerr << "[Error] Cannot open file " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate Offset (Each rank reads a unique chunk)
    // Row size = 4 bytes header + 1024 bytes data
    size_t row_size = 4 + d;
    size_t offset = (size_t)rank * n * row_size;
    
    // Skip training data (first 100k) to avoid overlap if needed, 
    // or just start from rank offset. Here we start from rank offset.
    // Safety check: ensure we don't seek past file end (assuming chunk_0000 is 200GB, plenty space)
    file.seekg(offset, std::ios::beg);

    std::vector<uint8_t> buffer(n * row_size);
    file.read(reinterpret_cast<char*>(buffer.data()), n * row_size);
    
    size_t read_count = file.gcount() / row_size;
    if (read_count < n && rank == 0) {
        std::cerr << "[Warning] File end reached. Requested " << n << ", got " << read_count << std::endl;
    }

    out_data.resize(read_count * d);
    
    #pragma omp parallel for
    for (size_t i = 0; i < read_count; ++i) {
        uint8_t* row_ptr = buffer.data() + i * row_size;
        uint8_t* vec_ptr = row_ptr + 4; // Skip 4-byte header
        for (int j = 0; j < d; ++j) {
            out_data[i * d + j] = static_cast<float>(vec_ptr[j]);
        }
    }
}

int main(int argc, char** argv) {
    // 1. MPI Init
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. Bind to GPU (Use Local Rank for safety)
    const char* local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = local_rank_env ? atoi(local_rank_env) : rank % 4;
    cudaSetDevice(local_rank);

    // ==========================================
    // Experiment Configuration
    // ==========================================
    int d = 1024; // DINO Dimension
    std::string file_path = "~/hpdic/data/dino10b/chunk_0000.bvecs";
    
    // We test different nlist configurations
    std::vector<int> nlist_list = {4096}; // 4096 is standard for 1M+ vectors
    
    // Load size per GPU (1M vectors = 4GB RAM, safe for P100)
    std::vector<size_t> nb_list = {1000000}; 

    // Prepare Data Buffers
    // We reuse this buffer for both rounds
    size_t max_nb = 1000000; 
    
    // Training data size (use a subset)
    size_t nt = 65536; 

    if (rank == 0) {
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "[MPI Scaling] Ranks: " << size << " | Local Rank Binding Checked" << std::endl;
        std::cout << "[Setup] Loading DINO 1024d Data (" << max_nb << " vectors/rank)..." << std::endl;
    }

    // 3. Load Data from Disk
    std::vector<float> all_xb;
    read_dino_part(file_path, rank, max_nb, d, all_xb);

    // Load Training Data (Rank 0 reads and broadcasts, or everyone reads same chunk)
    // To avoid MPI complexity in this snippet, everyone reads the FIRST 65k vectors for training
    std::vector<float> all_xt;
    read_dino_part(file_path, 0, nt, d, all_xt); // Rank offset 0 = start of file

    // Prepare IDs
    std::vector<idx_t> all_ids(max_nb);
    for (size_t i = 0; i < max_nb; ++i) all_ids[i] = (idx_t)(rank * max_nb + i);

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp

    // Common Config
    GpuIndexIVFFlatConfig config;
    config.device = local_rank;

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
            
            // --- Round 1: SIVF ---
            {
                // SIVF needs pre-allocated slab capacity
                // Capacity = Data + Buffer. 1.0M * 1.2 = 1.2M capacity
                size_t capacity = nb * 1.2; 
                
                GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
                index.initSlabManager(capacity, d); // Important: Pass 'd'
                
                // Train
                index.train(nt, all_xt.data());
                
                // Sync
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);

                // Benchmark Add
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

            // --- Round 2: Vanilla Faiss ---
            {
                // Re-init config to be safe
                GpuIndexIVFFlatConfig flatConfig;
                flatConfig.device = local_rank;
                
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