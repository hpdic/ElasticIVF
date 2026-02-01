/**
 * test_sivf_dino_search.cpp
 *
 * Evaluation: Distributed Search (Pareto Frontier: Recall vs QPS)
 * Dataset: DINO 10B (1024-dim, Subset)
 */

#include <mpi.h>
#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <map>

// Faiss Headers
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/GpuIndexFlat.h> // For GPU GT
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h> 

using namespace faiss;
using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Read DINO file (Absolute Path Required)
// ---------------------------------------------------------
void read_dino_file(const std::string& filename, size_t n, int d, std::vector<float>& out_buffer, size_t offset_vectors = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[Error] Cannot open file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t row_size = 4 + d; 
    size_t byte_offset = offset_vectors * row_size;
    file.seekg(byte_offset, std::ios::beg);

    std::vector<uint8_t> raw_bytes(n * row_size);
    file.read(reinterpret_cast<char*>(raw_bytes.data()), n * row_size);
    
    size_t read_count = file.gcount() / row_size;
    if (read_count < n) {
        out_buffer.resize(read_count * d); // Handle EOF gracefully
    } else {
        out_buffer.resize(n * d);
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < out_buffer.size() / d; ++i) {
        uint8_t* row_ptr = raw_bytes.data() + i * row_size;
        uint8_t* vec_ptr = row_ptr + 4; 
        for (int j = 0; j < d; ++j) {
            out_buffer[i * d + j] = static_cast<float>(vec_ptr[j]);
        }
    }
}

// ---------------------------------------------------------
// Helper: Compute Ground Truth on Rank 0 (GPU Accelerated)
// ---------------------------------------------------------
void compute_ground_truth(int d, size_t nb, const float* xb, 
                          size_t nq, const float* xq, 
                          std::vector<idx_t>& gt_ids, int k) {
    std::cout << "[GT] Computing Ground Truth for " << nb << " vectors on GPU..." << std::endl;
    
    // Resource wrapper
    faiss::gpu::StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); // 512MB Temp

    // Config to use device 0 (Rank 0's GPU)
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = 0; 

    // [FIX 1] Removed redundant faiss::METRIC_L2 argument
    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d, config);
    
    // 1. Add data to GPU (Copy host -> device)
    // 1M * 1024 * 4B = 4GB VRAM. P100 has 16GB, totally safe.
    index_flat.add(nb, xb); 
    
    std::vector<float> dists(nq * k);
    gt_ids.resize(nq * k);
    
    // 2. Brute-force Search on GPU
    index_flat.search(nq, xq, k, dists.data(), gt_ids.data());
    
    // Index is destroyed here, freeing the VRAM immediately
    std::cout << "[GT] Done. VRAM freed." << std::endl;
}

// ---------------------------------------------------------
// Helper: Merge Distributed Results (Map-Reduce style)
// ---------------------------------------------------------
void merge_results(int rank, int k, int nq, int world_size, 
                   const std::vector<float>& local_dists, 
                   const std::vector<idx_t>& local_ids,
                   std::vector<float>& global_dists, 
                   std::vector<idx_t>& global_ids) {
    
    // Gather all results to Rank 0
    std::vector<float> all_dists;
    std::vector<idx_t> all_ids;

    if (world_size > 1) {
        if (rank == 0) {
            all_dists.resize(world_size * nq * k);
            all_ids.resize(world_size * nq * k);
        }
        MPI_Gather(local_dists.data(), nq * k, MPI_FLOAT, 
                   all_dists.data(), nq * k, MPI_FLOAT, 
                   0, MPI_COMM_WORLD);
        MPI_Gather(local_ids.data(), nq * k, MPI_LONG, 
                   all_ids.data(), nq * k, MPI_LONG, 
                   0, MPI_COMM_WORLD);
    } else {
        all_dists = local_dists;
        all_ids = local_ids;
    }

    if (rank == 0) {
        global_dists.resize(nq * k);
        global_ids.resize(nq * k);

        // For each query, merge K results from `world_size` workers
        #pragma omp parallel for
        for (int q = 0; q < nq; ++q) {
            // Collect all (dist, id) pairs for this query
            std::vector<std::pair<float, idx_t>> candidates;
            candidates.reserve(world_size * k);
            
            for (int r = 0; r < world_size; ++r) {
                size_t base = (r * nq + q) * k;
                for (int i = 0; i < k; ++i) {
                    float d = all_dists[base + i];
                    idx_t id = all_ids[base + i];
                    if (id != -1) { // Filter invalid
                        candidates.push_back({d, id});
                    }
                }
            }

            // Sort by distance (ASC for L2)
            std::sort(candidates.begin(), candidates.end());

            // Take top k
            for (int i = 0; i < k && i < candidates.size(); ++i) {
                global_dists[q * k + i] = candidates[i].first;
                global_ids[q * k + i] = candidates[i].second;
            }
        }
    }
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
    int k = 10; // Recall@10
    
    // Absolute Paths
    std::string base_path = "/data/dino10b/chunk_0000.bvecs";
    std::string query_path = "/data/dino10b/queries_clean.bvecs"; 

    int nlist = 4096;
    // Safe memory limit: 100k per Rank = 1M Total
    size_t nb_per_rank = 100000; 
    size_t nq = 1000;            // Number of queries to test

    // 1. Load Data & Compute GT (Rank 0)
    std::vector<float> local_xb(nb_per_rank * d);
    std::vector<float> host_full_buffer;
    std::vector<float> queries(nq * d);
    std::vector<idx_t> gt_ids; // Global Ground Truth

    if (rank == 0) {
        // Load Base
        read_dino_file(base_path, nb_per_rank * size, d, host_full_buffer, 100000); // offset 100k
        
        // Load Queries (Check if exists first!)
        std::ifstream fq(query_path);
        if (!fq.good()) {
             // Fallback: use first 1000 vectors from base as queries if file missing
             std::cout << "[Warning] Query file not found. Using subset of base as queries." << std::endl;
             queries.resize(nq * d);
             memcpy(queries.data(), host_full_buffer.data(), nq * d * sizeof(float));
        } else {
             read_dino_file(query_path, nq, d, queries, 0);
        }
        
        // Compute GT (GPU)
        compute_ground_truth(d, nb_per_rank * size, host_full_buffer.data(), nq, queries.data(), gt_ids, k);
    }

    // Broadcast Queries to all
    MPI_Bcast(queries.data(), nq * d, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter Base Data
    MPI_Scatter(host_full_buffer.data(), nb_per_rank * d, MPI_FLOAT,
                local_xb.data(), nb_per_rank * d, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    // Free host buffer on Rank 0
    if (rank == 0) std::vector<float>().swap(host_full_buffer);

    // Local IDs (Global Offset)
    std::vector<idx_t> local_ids(nb_per_rank);
    for(size_t i=0; i<nb_per_rank; ++i) local_ids[i] = rank * nb_per_rank + i;

    // Train Data (Reuse local)
    size_t nt = 65536;
    std::vector<float> train_xt(nt * d);
    memcpy(train_xt.data(), local_xb.data(), nt * d * sizeof(float));

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024);
    GpuIndexIVFFlatConfig config;
    config.device = local_rank;

    // List of probes to test
    std::vector<int> probes = {1, 4, 8, 16, 32, 64};

    if (rank == 0) {
        printf("\n=================================================================================\n");
        printf("| %-8s | %-6s | %-12s | %-12s | %-12s |\n", "System", "nprobe", "Latency(ms)", "QPS", "Recall@10");
        printf("|----------|--------|--------------|--------------|--------------|\n");
    }

    // --- Benchmark Function ---
    auto run_benchmark = [&](const char* name, GpuIndexIVF* index) {
        index->train(nt, train_xt.data());
        index->add_with_ids(nb_per_rank, local_xb.data(), local_ids.data());
        
        // Sync before search phase
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int nprobe : probes) {
            // [FIX 2] Use direct member assignment instead of setter
            index->nprobe = nprobe; 
            
            std::vector<float> local_dists(nq * k);
            std::vector<idx_t> local_indices(nq * k);

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            
            index->search(nq, queries.data(), k, local_dists.data(), local_indices.data());
            
            cudaDeviceSynchronize();
            double t1 = MPI_Wtime();
            double local_lat = (t1 - t0);

            // Merge Results on Rank 0
            std::vector<float> global_dists;
            std::vector<idx_t> global_indices;
            merge_results(rank, k, nq, size, local_dists, local_indices, global_dists, global_indices);

            if (rank == 0) {
                // Calculate Recall
                int correct = 0;
                for (int q = 0; q < nq; ++q) {
                    std::vector<idx_t> gt_set;
                    for (int i=0; i<k; ++i) gt_set.push_back(gt_ids[q*k+i]);
                    
                    for (int i=0; i<k; ++i) {
                        idx_t res_id = global_indices[q*k+i];
                        for (idx_t truth : gt_set) {
                            if (res_id == truth) {
                                correct++;
                                break;
                            }
                        }
                    }
                }
                float recall = (float)correct / (nq * k);
                
                // QPS Calculation (Total Queries / Max Latency)
                double total_qps = nq / local_lat;

                printf("| %-8s | %-6d | %-12.2f | %-12.0f | %-12.4f |\n", 
                       name, nprobe, local_lat * 1000, total_qps, recall);
            }
        }
    };

    // --- Round 1: SIVF ---
    {
        size_t capacity = nb_per_rank * 1.5;
        GpuIndexSIVF sivf_index(&res, d, nlist, METRIC_L2, config);
        sivf_index.initSlabManager(capacity, d);
        run_benchmark("**SIVF**", &sivf_index);
    }

    // --- Round 2: Vanilla ---
    {
        GpuIndexIVFFlat vanilla_index(&res, d, nlist, METRIC_L2, config);
        run_benchmark("Vanilla", &vanilla_index);
    }

    MPI_Finalize();
    return 0;
}