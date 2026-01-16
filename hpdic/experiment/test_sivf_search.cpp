/**
 * faiss/hpdic/experiment/test_sivf_search.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Comprehensive Search Benchmark: SIVF vs Vanilla Faiss (IVFFlat)
 * Parameter Sweep: nb (Database Size) x nlist (Cluster Count) x nprobe
 *
 * This test evaluates the search throughput (QPS) and Recall@10 of the
 * Slab-based architecture against the standard contiguous memory implementation.
 */

#include <omp.h>
#include <sys/time.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// Faiss headers
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss;
using namespace faiss::gpu;

// Simple random number generation wrapper
float rand_float() {
    return (float)drand48();
}

int main() {
    // ==========================================
    // 1. Parameter Configuration
    // ==========================================
    std::vector<int> nb_list = {100000, 200000, 500000};
    std::vector<int> nlist_list = {1024, 4096, 16384};
    std::vector<int> nprobe_list = {10};

    int d = 128;
    int nq = 1000;
    int k = 10;

    printf("| %-10s | %-8s | %-6s | %-10s | %-10s | %-10s | %-8s |\n",
           "NB",
           "nlist",
           "nprobe",
           "System",
           "Add(s)",
           "SearchQPS",
           "Recall");
    printf("|------------|----------|--------|------------|------------|------------|----------|\n");

    StandardGpuResources res;
    res.noTempMemory();

    // [Fix] SIVF uses the basic GpuIndexIVFConfig
    GpuIndexIVFConfig sivf_config;
    sivf_config.device = 0;

    for (int nb : nb_list) {
        std::vector<float> xb(nb * d);
        std::vector<float> xq(nq * d);
        std::vector<long> ids(nb);

        srand48(42);
        for (long i = 0; i < nb; ++i) {
            ids[i] = i;
            for (int j = 0; j < d; ++j)
                xb[i * d + j] = rand_float();
        }

        // Generate queries by sampling from the base set
        for (int i = 0; i < nq; ++i) {
            int target = lrand48() % nb;
            for (int j = 0; j < d; ++j)
                xq[i * d + j] = xb[target * d + j];
        }

        for (int nlist : nlist_list) {
            for (int nprobe : nprobe_list) {
                // -------------------------------------------------
                // Round A: ElasticIVF (SIVF)
                // -------------------------------------------------
                {
                    size_t max_vectors = nb * 2L;
                    size_t slab_pool_size = nb * 2L;

                    // SIVF constructor accepts GpuIndexIVFConfig
                    GpuIndexSIVF sivf_index(
                            &res, d, nlist, METRIC_L2, sivf_config);
                    sivf_index.initSlabManager(max_vectors, slab_pool_size);
                    sivf_index.nprobe = nprobe;

                    // Train with a subset if nb is large
                    sivf_index.train(std::min((long)nb, 65536L), xb.data());

                    double t0 = omp_get_wtime();
                    sivf_index.add_with_ids(nb, xb.data(), ids.data());
                    double t_add = omp_get_wtime() - t0;

                    // Warmup Search
                    {
                        std::vector<float> D(nq * k);
                        std::vector<long> I(nq * k);
                        sivf_index.search(nq, xq.data(), k, D.data(), I.data());
                    }

                    // Benchmark Search
                    std::vector<float> D(nq * k);
                    std::vector<long> I(nq * k);

                    t0 = omp_get_wtime();
                    sivf_index.search(nq, xq.data(), k, D.data(), I.data());
                    double t_search = omp_get_wtime() - t0;
                    double qps = nq / t_search;

                    // Calculate Recall@K
                    int correct = 0;
                    for (int i = 0; i < nq; ++i)
                        if (D[i * k] < 1e-4) // Assuming exact match distance ~ 0
                            correct++;
                    float recall = 100.0f * correct / nq;

                    printf("| %-10d | %-8d | %-6d | %-10s | %-10.4f | %-10.0f | %-6.1f%% |\n",
                           nb,
                           nlist,
                           nprobe,
                           "**SIVF**",
                           t_add,
                           qps,
                           recall);
                }

                // -------------------------------------------------
                // Round B: Vanilla Faiss (Baseline)
                // -------------------------------------------------
                {
                    // 1. CPU Train (Standard Faiss workflow)
                    IndexFlatL2 cpu_quantizer(d);
                    IndexIVFFlat cpu_index(&cpu_quantizer, d, nlist, METRIC_L2);
                    cpu_index.train(std::min((long)nb, 65536L), xb.data());

                    // [Fix] Must use GpuIndexIVFFlatConfig specifically
                    GpuIndexIVFFlatConfig flat_config;
                    flat_config.device = 0;

                    // 2. GPU Index Construction (Load from CPU index)
                    GpuIndexIVFFlat gpu_index(
                            &res, d, nlist, METRIC_L2, flat_config);
                    gpu_index.copyFrom(&cpu_index);

                    // 3. Set Parameters
                    gpu_index.nprobe = nprobe;

                    double t0 = omp_get_wtime();
                    gpu_index.add_with_ids(nb, xb.data(), ids.data());
                    double t_add = omp_get_wtime() - t0;

                    // Warmup
                    {
                        std::vector<float> D(nq * k);
                        std::vector<long> I(nq * k);
                        gpu_index.search(nq, xq.data(), k, D.data(), I.data());
                    }

                    // Benchmark Search
                    std::vector<float> D(nq * k);
                    std::vector<long> I(nq * k);

                    t0 = omp_get_wtime();
                    gpu_index.search(nq, xq.data(), k, D.data(), I.data());
                    double t_search = omp_get_wtime() - t0;
                    double qps = nq / t_search;

                    int correct = 0;
                    for (int i = 0; i < nq; ++i)
                        if (D[i * k] < 1e-4)
                            correct++;
                    float recall = 100.0f * correct / nq;

                    printf("| %-10s | %-8s | %-6s | %-10s | %-10.4f | %-10.0f | %-6.1f%% |\n",
                           "\"",
                           "\"",
                           "\"",
                           "Vanilla",
                           t_add,
                           qps,
                           recall);

                } // gpu_index is automatically destructed here, freeing VRAM

                fflush(stdout);
            }
        }
    }
    return 0;
}

/** Example output:
 *
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ cd ~/ElasticIVF/build
rm -f faiss/gpu/CMakeFiles/faiss_gpu_objs.dir/impl/SIVFAppend.cu.o
rm -f faiss/gpu/CMakeFiles/faiss_gpu_objs.dir/impl/SIVFSearch.cu.o
make test_sivf_search -j
./faiss/gpu/test_sivf_search
[  0%] Building CUDA object faiss/gpu/CMakeFiles/faiss_gpu_objs.dir/impl/SIVFAppend.cu.o
[  0%] Building CUDA object faiss/gpu/CMakeFiles/faiss_gpu_objs.dir/impl/SIVFSearch.cu.o
[ 64%] Built target faiss_gpu_objs
[ 64%] Linking CXX static library libfaiss.a
[ 97%] Built target faiss
[100%] Linking CXX executable test_sivf_search
[100%] Built target test_sivf_search
| NB         | nlist    | nprobe | System     | Add(s)     | SearchQPS  | Recall   |
|------------|----------|--------|------------|------------|------------|----------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200000 -> 200000
  > Data Buffer: 200000 -> 6400000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.38 s, search 0.27 s): objective=617502 imbalance=1.203 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 100000     | 1024     | 10     | **SIVF**   | 0.0176     | 265258     | 100.0 % |
| "          | "        | "      | Vanilla    | 0.0658     | 495769     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200000 -> 200000
  > Data Buffer: 200000 -> 6400000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 18 (0.45 s, search 0.32 s): objective=568760 imbalance=1.940 nsplit=0       
  Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 100000     | 4096     | 10     | **SIVF**   | 0.0257     | 382763     | 100.0 % |
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| "          | "        | "      | Vanilla    | 0.1166     | 560801     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200000 -> 200000
  > Data Buffer: 200000 -> 6400000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 16384 centroids: please provide at least 638976 training points
Clustering 65536 points in 128D to 16384 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 10 (0.59 s, search 0.45 s): objective=425710 imbalance=2.634 nsplit=0       
  Converged at iteration 10: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 100000     | 16384    | 10     | **SIVF**   | 0.0621     | 295321     | 100.0 % |
WARNING clustering 65536 points to 16384 centroids: please provide at least 638976 training points
| "          | "        | "      | Vanilla    | 0.3678     | 322907     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   400000 -> 400000
  > Data Buffer: 400000 -> 12800000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.33 s, search 0.24 s): objective=617502 imbalance=1.203 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 200000     | 1024     | 10     | **SIVF**   | 0.0335     | 157573     | 100.0 % |
| "          | "        | "      | Vanilla    | 0.0958     | 352334     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   400000 -> 400000
  > Data Buffer: 400000 -> 12800000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 18 (0.43 s, search 0.31 s): objective=568760 imbalance=1.940 nsplit=0       
  Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 200000     | 4096     | 10     | **SIVF**   | 0.0486     | 263014     | 100.0 % |
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| "          | "        | "      | Vanilla    | 0.1601     | 443137     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   400000 -> 400000
  > Data Buffer: 400000 -> 12800000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 16384 centroids: please provide at least 638976 training points
Clustering 65536 points in 128D to 16384 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 10 (0.59 s, search 0.45 s): objective=425710 imbalance=2.634 nsplit=0       
  Converged at iteration 10: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 200000     | 16384    | 10     | **SIVF**   | 0.1194     | 205614     | 100.0 % |
WARNING clustering 65536 points to 16384 centroids: please provide at least 638976 training points
| "          | "        | "      | Vanilla    | 0.4504     | 320675     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1000000 -> 1000000
  > Data Buffer: 1000000 -> 32000000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.31 s, search 0.23 s): objective=617502 imbalance=1.203 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 500000     | 1024     | 10     | **SIVF**   | 0.0802     | 69784      | 100.0 % |
| "          | "        | "      | Vanilla    | 0.2163     | 176131     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1000000 -> 1000000
  > Data Buffer: 1000000 -> 32000000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
Clustering 65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 18 (0.42 s, search 0.31 s): objective=568760 imbalance=1.940 nsplit=0       
  Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 500000     | 4096     | 10     | **SIVF**   | 0.1187     | 117890     | 100.0 % |
WARNING clustering 65536 points to 4096 centroids: please provide at least 159744 training points
| "          | "        | "      | Vanilla    | 0.2980     | 227213     | 100.0 % |

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   1000000 -> 1000000
  > Data Buffer: 1000000 -> 32000000 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 65536 points to 16384 centroids: please provide at least 638976 training points
Clustering 65536 points in 128D to 16384 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 10 (0.56 s, search 0.43 s): objective=425710 imbalance=2.634 nsplit=0       
  Converged at iteration 10: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 500000     | 16384    | 10     | **SIVF**   | 0.2939     | 100102     | 100.0 % |
WARNING clustering 65536 points to 16384 centroids: please provide at least 638976 training points
| "          | "        | "      | Vanilla    | 0.7726     | 213205     | 100.0 % |
cc@rtx6000:~/ElasticIVF/build$ 
 */