/**
 * test_sivf_dino_search.cpp
 *
 * Evaluation: Distributed Search (Pareto Frontier: Recall vs QPS)
 * Dataset: DINO 10B (1024-dim, Subset)
 * * Improvements:
 * 1. Command-line arguments for nlist, nb, and probes.
 * 2. GPU Ground Truth calculation.
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
#include <string>
#include <sstream>

// Faiss Headers
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/GpuIndexFlat.h> 
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h> 

using namespace faiss;
using namespace faiss::gpu;

// ---------------------------------------------------------
// Helper: Parse Command Line Arguments
// ---------------------------------------------------------
struct Args {
    int nlist = 8192;
    size_t nb_per_rank = 100000;
    std::vector<int> probes = {8, 16, 32, 64, 128};
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--nlist") {
            if (i + 1 < argc) args.nlist = std::stoi(argv[++i]);
        } else if (arg == "--nb") {
            if (i + 1 < argc) args.nb_per_rank = std::stoll(argv[++i]);
        } else if (arg == "--probes") {
            if (i + 1 < argc) {
                std::string list = argv[++i];
                std::stringstream ss(list);
                std::string item;
                args.probes.clear();
                while (std::getline(ss, item, ',')) {
                    args.probes.push_back(std::stoi(item));
                }
            }
        }
    }
    return args;
}

// ---------------------------------------------------------
// Helper: Read DINO file
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
        out_buffer.resize(read_count * d); 
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
    
    faiss::gpu::StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 

    faiss::gpu::GpuIndexFlatConfig config;
    config.device = 0; 

    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d, config);
    index_flat.add(nb, xb); 
    
    std::vector<float> dists(nq * k);
    gt_ids.resize(nq * k);
    
    index_flat.search(nq, xq, k, dists.data(), gt_ids.data());
    std::cout << "[GT] Done. VRAM freed." << std::endl;
}

// ---------------------------------------------------------
// Helper: Merge Distributed Results
// ---------------------------------------------------------
void merge_results(int rank, int k, int nq, int world_size, 
                   const std::vector<float>& local_dists, 
                   const std::vector<idx_t>& local_ids,
                   std::vector<float>& global_dists, 
                   std::vector<idx_t>& global_ids) {
    
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

        #pragma omp parallel for
        for (int q = 0; q < nq; ++q) {
            std::vector<std::pair<float, idx_t>> candidates;
            candidates.reserve(world_size * k);
            
            for (int r = 0; r < world_size; ++r) {
                size_t base = (r * nq + q) * k;
                for (int i = 0; i < k; ++i) {
                    float d = all_dists[base + i];
                    idx_t id = all_ids[base + i];
                    if (id != -1) candidates.push_back({d, id});
                }
            }
            std::sort(candidates.begin(), candidates.end());

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

    Args args = parse_args(argc, argv);

    const char* local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = local_rank_env ? atoi(local_rank_env) : rank % 4;
    cudaSetDevice(local_rank);

    // ==========================================
    // Config
    // ==========================================
    int d = 1024;
    int k = 10; 
    std::string base_path = "/home/cc/hpdic/data/dino10b/chunk_0000.bvecs";
    std::string query_path = "/home/cc/hpdic/data/dino10b/queries_clean.bvecs"; 

    int nlist = args.nlist;
    size_t nb_per_rank = args.nb_per_rank;
    size_t nq = 1000;            

    if (rank == 0) {
        printf("[Config] nlist=%d, nb_per_rank=%ld, probes=[", nlist, nb_per_rank);
        for(size_t i=0; i<args.probes.size(); ++i) printf("%d%s", args.probes[i], i==args.probes.size()-1?"":",");
        printf("]\n");
    }

    // 1. Load Data & Compute GT (Rank 0)
    std::vector<float> local_xb(nb_per_rank * d);
    std::vector<float> host_full_buffer;
    std::vector<float> queries(nq * d);
    std::vector<idx_t> gt_ids; 

    if (rank == 0) {
        read_dino_file(base_path, nb_per_rank * size, d, host_full_buffer, 100000); 
        
        std::ifstream fq(query_path);
        if (!fq.good()) {
             std::cout << "[Warning] Query file not found. Using subset." << std::endl;
             queries.resize(nq * d);
             memcpy(queries.data(), host_full_buffer.data(), nq * d * sizeof(float));
        } else {
             read_dino_file(query_path, nq, d, queries, 0);
        }
        compute_ground_truth(d, nb_per_rank * size, host_full_buffer.data(), nq, queries.data(), gt_ids, k);
    }

    MPI_Bcast(queries.data(), nq * d, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatter(host_full_buffer.data(), nb_per_rank * d, MPI_FLOAT,
                local_xb.data(), nb_per_rank * d, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) std::vector<float>().swap(host_full_buffer);

    std::vector<idx_t> local_ids(nb_per_rank);
    for(size_t i=0; i<nb_per_rank; ++i) local_ids[i] = rank * nb_per_rank + i;

    // Use ALL local data for training
    size_t nt = nb_per_rank; 

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024);
    GpuIndexIVFFlatConfig config;
    config.device = local_rank;

    if (rank == 0) {
        printf("\n=================================================================================\n");
        printf("| %-8s | %-6s | %-12s | %-12s | %-12s |\n", "System", "nprobe", "Latency(ms)", "QPS", "Recall@10");
        printf("|----------|--------|--------------|--------------|--------------|\n");
    }

    auto run_benchmark = [&](const char* name, GpuIndexIVF* index) {
        index->train(nt, local_xb.data());
        index->add_with_ids(nb_per_rank, local_xb.data(), local_ids.data());
        
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int nprobe : args.probes) {
            index->nprobe = nprobe; 
            
            std::vector<float> local_dists(nq * k);
            std::vector<idx_t> local_indices(nq * k);

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            
            index->search(nq, queries.data(), k, local_dists.data(), local_indices.data());
            
            cudaDeviceSynchronize();
            double t1 = MPI_Wtime();
            double local_lat = (t1 - t0);

            std::vector<float> global_dists;
            std::vector<idx_t> global_indices;
            merge_results(rank, k, nq, size, local_dists, local_indices, global_dists, global_indices);

            if (rank == 0) {
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
                double total_qps = nq / local_lat;

                printf("| %-8s | %-6d | %-12.2f | %-12.0f | %-12.4f |\n", 
                       name, nprobe, local_lat * 1000, total_qps, recall);
            }
        }
    };

    // --- Round 1: SIVF ---
    {
        size_t capacity = nb_per_rank * 1.5;
        GpuIndexSIVF sivf_index(&res, d, args.nlist, METRIC_L2, config);
        sivf_index.initSlabManager(capacity, d);
        run_benchmark("**SIVF**", &sivf_index);
    }

    // --- Round 2: Vanilla ---
    {
        GpuIndexIVFFlat vanilla_index(&res, d, args.nlist, METRIC_L2, config);
        run_benchmark("Vanilla", &vanilla_index);
    }

    MPI_Finalize();
    return 0;
}

/** Example output:
 * 
cc@p100x2:~$ cat << 'EOF' > run_gpu3.sh
#!/bin/bash
EXE_PATH="/home/cc/hpdic/ElasticIVF/build/test_sivf_dino_search"
NLIST=30000
NB=200000
PROBES="4,8,16,32,64,128"

echo "-----------------------------------------------------------------------"
echo ">>> STARTING LOCAL TEST ON GPU3: nlist=$NLIST | nb=$NB"
echo "-----------------------------------------------------------------------"

mpirun --allow-run-as-root \
    -np 2 \
    --host localhost:2 \
    -x LD_LIBRARY_PATH \
    $EXE_PATH \
    --nlist $NLIST \
    --nb $NB \
    --probes $PROBES
EOF
cc@p100x2:~$ chmod +x run_gpu3.sh 
cc@p100x2:~$ ./run_gpu3.sh
-----------------------------------------------------------------------
>>> STARTING LOCAL TEST ON GPU3: nlist=30000 | nb=200000
-----------------------------------------------------------------------
[Config] nlist=30000, nb_per_rank=200000, probes=[4,8,16,32,64,128]
[Warning] Query file not found. Using subset.
[GT] Computing Ground Truth for 400000 vectors on GPU...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[GT] Done. VRAM freed.
[HPDIC MOD] Faiss GPU initialized on device ID: 1

=================================================================================
| System   | nprobe | Latency(ms)  | QPS          | Recall@10    |
|----------|--------|--------------|--------------|--------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 50971
  > Data Buffer: 300000 -> 1631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1024 -> 50971
  > Data Buffer: 300000 -> 1631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 200000 points in 1024D to 30000 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.84 s
WARNING clustering 200000 points to 30000 centroids: please provide at least 1170000 training points
WARNING clustering 200000 points to 30000 centroids: please provide at least 1170000 training points
Clustering 200000 points in 1024D to 30000 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.83 s
  Iteration 14 (32.95 s, search 23.45 s): objective=3.13129e+10 imbalance=1.519 nsplit=0       
  Converged at iteration 14: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 30000 centroids.
  Iteration 14 (33.62 s, search 23.59 s): objective=3.13028e+10 imbalance=1.529 nsplit=0       
  Converged at iteration 14: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 30000 centroids.
| **SIVF** | 4      | 13.26        | 75423        | 0.7116       |
| **SIVF** | 8      | 13.46        | 74274        | 0.8160       |
| **SIVF** | 16     | 17.53        | 57057        | 0.8925       |
| **SIVF** | 32     | 25.52        | 39190        | 0.9442       |
| **SIVF** | 64     | 41.30        | 24211        | 0.9741       |
| **SIVF** | 128    | 70.42        | 14200        | 0.9895       |
WARNING clustering 200000 points to 30000 centroids: please provide at least 1170000 training points
WARNING clustering 200000 points to 30000 centroids: please provide at least 1170000 training points
| Vanilla  | 4      | 11.70        | 85483        | 0.7116       |
| Vanilla  | 8      | 12.85        | 77842        | 0.8160       |
| Vanilla  | 16     | 16.51        | 60577        | 0.8925       |
| Vanilla  | 32     | 24.08        | 41529        | 0.9442       |
| Vanilla  | 64     | 39.16        | 25536        | 0.9741       |
| Vanilla  | 128    | 69.49        | 14391        | 0.9895       |
cc@p100x2:~$ 
 */