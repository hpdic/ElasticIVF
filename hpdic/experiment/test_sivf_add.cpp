/**
 * faiss/hpdic/experiment/test_sivf_add.cpp
 * Comprehensive Benchmark: ElasticIVF vs Vanilla Faiss
 * Parameter Sweep: nb (Database Size) x nlist (Cluster Count)
 */

#include <sys/time.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss;
using namespace faiss::gpu;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 快速随机数生成 (避免 huge loop 耗时太久)
void generate_data(size_t n, int d, std::vector<float>& data) {
    // 只生成前 1000 个随机数，后面循环拷贝，加快 benchmark 准备时间
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

int main() {
    // ==========================================
    // 实验参数配置 (Parameter Sweep)
    // ==========================================
    int d = 128;

    // 遍历不同的 nlist (聚类中心数)
    std::vector<int> nlist_list = {1024, 4096, 16384};

    // 遍历不同的数据库大小 (从 100万 到 1000万)
    // 注意：10M * 128 * 4B = 5GB 显存，RTX 6000 轻松吃下
    std::vector<size_t> nb_list = {1000000, 5000000, 10000000};

    // 最大的 nb，用于预先生成数据
    size_t max_nb = 10000000;
    size_t max_nt = 256 * 1024; // 足够大的训练集

    printf("Preparing Data (Max NB=%ld, Max NT=%ld)...\n", max_nb, max_nt);
    std::vector<float> all_xb(max_nb * d);
    generate_data(max_nb, d, all_xb);

    std::vector<float> all_xt(max_nt * d);
    generate_data(max_nt, d, all_xt);

    std::vector<idx_t> all_ids(max_nb);
    for (size_t i = 0; i < max_nb; ++i)
        all_ids[i] = i;

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024); // 1GB Temp Memory

    GpuIndexIVFConfig config;
    config.device = 0;

    // 输出表头
    printf("\n| %-10s | %-10s | %-15s | %-10s | %-15s | %-10s |\n",
           "NB",
           "nlist",
           "System",
           "Time(s)",
           "QPS (vec/s)",
           "Speedup");
    printf("|------------|------------|-----------------|------------|-----------------|------------|\n");

    // ==========================================
    // Loop
    // ==========================================
    for (size_t nb : nb_list) {
        for (int nlist : nlist_list) {
            // 动态计算需要的训练数据量 (39 * nlist 是 Faiss 推荐值)
            size_t nt = std::max((size_t)65536, (size_t)nlist * 40);
            if (nt > max_nt)
                nt = max_nt;

            double sivf_qps = 0;
            double vanilla_qps = 0;

            // --- Round 1: SIVF ---
            {
                size_t max_vectors = nb * 2;
                size_t slab_pool_size =
                        max_vectors / 32 + (nlist * 2); // 稍微多给点 redundant

                GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
                index.initSlabManager(max_vectors, slab_pool_size);

                // Train
                index.train(nt, all_xt.data());

                // Add
                cudaDeviceSynchronize();
                double t0 = elapsed();
                index.add_with_ids(nb, all_xb.data(), all_ids.data());
                cudaDeviceSynchronize();
                double t1 = elapsed();

                double time_cost = t1 - t0;
                sivf_qps = nb / time_cost;

                printf("| %-10ld | %-10d | %-15s | %-10.4f | %-15.0f | %-10s |\n",
                       nb,
                       nlist,
                       "**SIVF**",
                       time_cost,
                       sivf_qps,
                       "-");
            }

            // --- Round 2: Vanilla ---
            {
                GpuIndexIVFFlatConfig flatConfig;
                flatConfig.device = 0;
                faiss::gpu::GpuIndexIVFFlat index(
                        &res, d, nlist, METRIC_L2, flatConfig);

                // Train
                index.train(nt, all_xt.data());

                // Add
                cudaDeviceSynchronize();
                double t0 = elapsed();
                index.add_with_ids(nb, all_xb.data(), all_ids.data());
                cudaDeviceSynchronize();
                double t1 = elapsed();

                double time_cost = t1 - t0;
                vanilla_qps = nb / time_cost;

                printf("| %-10s | %-10s | %-15s | %-10.4f | %-15.0f | %-10.2fx |\n",
                       "\"",
                       "\"",
                       "Vanilla",
                       time_cost,
                       vanilla_qps,
                       sivf_qps / vanilla_qps);
            }
            // 分隔线
            printf("|------------|------------|-----------------|------------|-----------------|------------|\n");
        }
    }

    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_add
Preparing Data (Max NB=10000000, Max NT=262144)...

| NB         | nlist      | System          | Time(s)    | QPS (vec/s)     |
Speedup    |
|------------|------------|-----------------|------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 11 (0.22 s, search 0.15 s): objective=553137 imbalance=1.923
nsplit=0 Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 1000000    | 1024       | **SIVF**        | 0.1731     | 5777175         | - |
| "          | "          | Vanilla         | 0.4799     | 2083846 | 2.77      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 7 (0.61 s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 1000000    | 4096       | **SIVF**        | 0.2323     | 4304000         | - |
| "          | "          | Vanilla         | 0.5991     | 1669100 | 2.58      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points Clustering 262144 points in 128D to 16384 clusters, redo
1 times, 20 iterations Preprocessing in 0.14 s Iteration 19 (66.93 s,
search 3.34 s): objective=1413.36 imbalance=1.641 nsplit=6390 [SIVF::train] GPU
K-Means complete. Quantizer populated with 16384 centroids. | 1000000    | 16384
| **SIVF**        | 0.7663     | 1305052         | -          | WARNING
clustering 262144 points to 16384 centroids: please provide at least 638976
training points | "          | "          | Vanilla         | 1.5553     |
642968          | 2.03      x |
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 11 (0.18 s, search 0.12 s): objective=553137 imbalance=1.923
nsplit=0 Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 5000000    | 1024       | **SIVF**        | 1.4775     | 3384067         | - |
| "          | "          | Vanilla         | 2.1719     | 2302144 | 1.47      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 7 (0.59 s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 5000000    | 4096       | **SIVF**        | 1.1526     | 4338003         | - |
| "          | "          | Vanilla         | 2.7086     | 1845942 | 2.35      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points Clustering 262144 points in 128D to 16384 clusters, redo
1 times, 20 iterations Preprocessing in 0.12 s Iteration 19 (67.36 s,
search 3.33 s): objective=1413.36 imbalance=1.641 nsplit=6390 [SIVF::train] GPU
K-Means complete. Quantizer populated with 16384 centroids. | 5000000    | 16384
| **SIVF**        | 3.7168     | 1345241         | -          | WARNING
clustering 262144 points to 16384 centroids: please provide at least 638976
training points | "          | "          | Vanilla         | 6.3567     |
786575          | 1.71      x |
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 11 (0.25 s, search 0.13 s): objective=553137 imbalance=1.923
nsplit=0 Converged at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 10000000   | 1024       | **SIVF**        | 2.4288     | 4117185         | - |
| "          | "          | Vanilla         | 4.3385     | 2304946 | 1.79      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.08 s
  Iteration 7 (0.64 s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2
  Converged at iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 10000000   | 4096       | **SIVF**        | 2.4919     | 4012971         | - |
| "          | "          | Vanilla         | 5.4752     | 1826413 | 2.20      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points Clustering 262144 points in 128D to 16384 clusters, redo
1 times, 20 iterations Preprocessing in 0.11 s Iteration 19 (66.99 s,
search 3.32 s): objective=1413.36 imbalance=1.641 nsplit=6390 [SIVF::train] GPU
K-Means complete. Quantizer populated with 16384 centroids. | 10000000   | 16384
| **SIVF**        | 7.0031     | 1427939         | -          | WARNING
clustering 262144 points to 16384 centroids: please provide at least 638976
training points | "          | "          | Vanilla         | 14.7557    |
677705          | 2.11      x |
|------------|------------|-----------------|------------|-----------------|------------|
cc@rtx6000:~/ElasticIVF/build$
 */