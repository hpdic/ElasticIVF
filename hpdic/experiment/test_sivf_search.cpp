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

// 简单的随机数生成
float rand_float() {
    return (float)drand48();
}

int main() {
    // ==========================================
    // 1. 参数配置区
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

    // [修正] SIVF 使用基础 Config
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

                    // SIVF 构造函数接受 GpuIndexIVFConfig
                    GpuIndexSIVF sivf_index(
                            &res, d, nlist, METRIC_L2, sivf_config);
                    sivf_index.initSlabManager(max_vectors, slab_pool_size);
                    sivf_index.nprobe = nprobe;

                    sivf_index.train(std::min((long)nb, 65536L), xb.data());

                    double t0 = omp_get_wtime();
                    sivf_index.add_with_ids(nb, xb.data(), ids.data());
                    double t_add = omp_get_wtime() - t0;

                    // Warmup
                    {
                        std::vector<float> D(nq * k);
                        std::vector<long> I(nq * k);
                        sivf_index.search(nq, xq.data(), k, D.data(), I.data());
                    }

                    std::vector<float> D(nq * k);
                    std::vector<long> I(nq * k);

                    t0 = omp_get_wtime();
                    sivf_index.search(nq, xq.data(), k, D.data(), I.data());
                    double t_search = omp_get_wtime() - t0;
                    double qps = nq / t_search;

                    int correct = 0;
                    for (int i = 0; i < nq; ++i)
                        if (D[i * k] < 1e-4)
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
                    // 1. CPU Train
                    IndexFlatL2 cpu_quantizer(d);
                    IndexIVFFlat cpu_index(&cpu_quantizer, d, nlist, METRIC_L2);
                    cpu_index.train(std::min((long)nb, 65536L), xb.data());

                    // [修正] 这里必须使用 GpuIndexIVFFlatConfig
                    GpuIndexIVFFlatConfig flat_config;
                    flat_config.device = 0;

                    // 2. GPU Index Construction
                    GpuIndexIVFFlat gpu_index(
                            &res, d, nlist, METRIC_L2, flat_config);
                    gpu_index.copyFrom(&cpu_index);

                    // 3. Set Params
                    gpu_index.nprobe = nprobe;

                    double t0 = omp_get_wtime();
                    gpu_index.add_with_ids(nb, xb.data(), ids.data());
                    double t_add = omp_get_wtime() - t0;

                    {
                        std::vector<float> D(nq * k);
                        std::vector<long> I(nq * k);
                        gpu_index.search(nq, xq.data(), k, D.data(), I.data());
                    }

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

                } // gpu_index 自动析构

                fflush(stdout);
            }
        }
    }
    return 0;
}

/** Example output:
 *
 cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_search
| NB         | nlist    | nprobe | System     | Add(s)     | SearchQPS  | Recall
|
|------------|----------|--------|------------|------------|------------|----------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 19 (0.39 s, search 0.27 s): objective=617502 imbalance=1.203
nsplit=0 [SIVF::train] GPU K-Means complete. Quantizer populated with 1024
centroids. | 100000     | 1024     | 10     | **SIVF**   | 0.0319     | 280611
| 100.0 % | | "          | "        | "      | Vanilla    | 0.0614     | 529435
| 100.0 % | [SIVF::train] WARNING: Base train failed. Executing GPU K-Means
fallback... WARNING clustering 65536 points to 4096 centroids: please provide at
least 159744 training points Clustering 65536 points in 128D to 4096 clusters,
redo 1 times, 20 iterations Preprocessing in 0.03 s Iteration 18 (0.45 s, search
0.32 s): objective=568760 imbalance=1.940 nsplit=0 Converged at iteration 18:
objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 100000     | 4096     | 10     | **SIVF**   | 0.0347     | 417376     | 100.0
% | WARNING clustering 65536 points to 4096 centroids: please provide at least
159744 training points | "          | "        | "      | Vanilla    | 0.1138 |
568740     | 100.0 % | [SIVF::train] WARNING: Base train failed. Executing GPU
K-Means fallback... WARNING clustering 65536 points to 16384 centroids: please
provide at least 638976 training points Clustering 65536 points in 128D to 16384
clusters, redo 1 times, 20 iterations Preprocessing in 0.03 s Iteration 10 (0.60
s, search 0.46 s): objective=425710 imbalance=2.634 nsplit=0 Converged at
iteration 10: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 100000     | 16384    | 10     | **SIVF**   | 0.0687     | 310710     | 100.0
% | WARNING clustering 65536 points to 16384 centroids: please provide at least
638976 training points | "          | "        | "      | Vanilla    | 0.3600 |
326045     | 100.0 % | [SIVF::train] WARNING: Base train failed. Executing GPU
K-Means fallback... Clustering 65536 points in 128D to 1024 clusters, redo 1
times, 20 iterations Preprocessing in 0.03 s Iteration 19 (0.34 s, search 0.24
s): objective=617502 imbalance=1.203 nsplit=0 [SIVF::train] GPU K-Means
complete. Quantizer populated with 1024 centroids. | 200000     | 1024     | 10
| **SIVF**   | 0.0507     | 166108     | 100.0 % | | "          | "        | "
| Vanilla    | 0.0986     | 352114     | 100.0 % | [SIVF::train] WARNING: Base
train failed. Executing GPU K-Means fallback... WARNING clustering 65536 points
to 4096 centroids: please provide at least 159744 training points Clustering
65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations Preprocessing
in 0.03 s Iteration 18 (0.43 s, search 0.32 s): objective=568760 imbalance=1.940
nsplit=0 Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 200000     | 4096     | 10     | **SIVF**   | 0.0610     | 272699     | 100.0
% | WARNING clustering 65536 points to 4096 centroids: please provide at least
159744 training points | "          | "        | "      | Vanilla    | 0.1563 |
446427     | 100.0 % | [SIVF::train] WARNING: Base train failed. Executing GPU
K-Means fallback... WARNING clustering 65536 points to 16384 centroids: please
provide at least 638976 training points Clustering 65536 points in 128D to 16384
clusters, redo 1 times, 20 iterations Preprocessing in 0.03 s Iteration 10 (0.58
s, search 0.44 s): objective=425710 imbalance=2.634 nsplit=0 Converged at
iteration 10: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 200000     | 16384    | 10     | **SIVF**   | 0.1280     | 216029     | 100.0
% | WARNING clustering 65536 points to 16384 centroids: please provide at least
638976 training points | "          | "        | "      | Vanilla    | 0.4637 |
309570     | 100.0 % | [SIVF::train] WARNING: Base train failed. Executing GPU
K-Means fallback... Clustering 65536 points in 128D to 1024 clusters, redo 1
times, 20 iterations Preprocessing in 0.03 s Iteration 19 (0.32 s, search 0.23
s): objective=617502 imbalance=1.203 nsplit=0 [SIVF::train] GPU K-Means
complete. Quantizer populated with 1024 centroids. | 500000     | 1024     | 10
| **SIVF**   | 0.1023     | 71866      | 100.0 % | | "          | "        | "
| Vanilla    | 0.2130     | 177202     | 100.0 % | [SIVF::train] WARNING: Base
train failed. Executing GPU K-Means fallback... WARNING clustering 65536 points
to 4096 centroids: please provide at least 159744 training points Clustering
65536 points in 128D to 4096 clusters, redo 1 times, 20 iterations Preprocessing
in 0.03 s Iteration 18 (0.45 s, search 0.31 s): objective=568760 imbalance=1.940
nsplit=0 Converged at iteration 18: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 500000     | 4096     | 10     | **SIVF**   | 0.1370     | 111601     | 100.0
% | WARNING clustering 65536 points to 4096 centroids: please provide at least
159744 training points | "          | "        | "      | Vanilla    | 0.2925 |
166162     | 100.0 % | [SIVF::train] WARNING: Base train failed. Executing GPU
K-Means fallback... WARNING clustering 65536 points to 16384 centroids: please
provide at least 638976 training points Clustering 65536 points in 128D to 16384
clusters, redo 1 times, 20 iterations Preprocessing in 0.03 s Iteration 10 (0.57
s, search 0.44 s): objective=425710 imbalance=2.634 nsplit=0 Converged at
iteration 10: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 500000     | 16384    | 10     | **SIVF**   | 0.3059     | 104455     | 100.0
% | WARNING clustering 65536 points to 16384 centroids: please provide at least
638976 training points | "          | "        | "      | Vanilla    | 0.7557 |
214485     | 100.0 % | cc@rtx6000:~/ElasticIVF/build$
 */