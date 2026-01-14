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
  Iteration 0 (0.05 s, search 0.03 s): objective=8  Iteration 1 (0.07 s, search
0.04 s): objective=5  Iteration 2 (0.08 s, search 0.05 s): objective=5 Iteration
3 (0.10 s, search 0.06 s): objective=5  Iteration 4 (0.11 s, search 0.07 s):
objective=5  Iteration 5 (0.13 s, search 0.08 s): objective=5  Iteration 6 (0.14
s, search 0.10 s): objective=5  Iteration 7 (0.16 s, search 0.11 s): objective=5
Iteration 8 (0.18 s, search 0.12 s): objective=5  Iteration 9 (0.19 s, search
0.13 s): objective=5  Iteration 10 (0.21 s, search 0.14 s): objective= Iteration
11 (0.22 s, search 0.15 s): objective=553137 imbalance=1.923 nsplit=0 Converged
at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 1000000    | 1024       | **SIVF**        | 0.1636     | 6111348         | - |
| "          | "          | Vanilla         | 0.4774     | 2094775 | 2.92      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 0 (0.14 s, search 0.04 s): objective=1  Iteration 1 (0.23 s, search
0.08 s): objective=9  Iteration 2 (0.30 s, search 0.12 s): objective=8 Iteration
3 (0.36 s, search 0.16 s): objective=8  Iteration 4 (0.42 s, search 0.19 s):
objective=8  Iteration 5 (0.49 s, search 0.23 s): objective=8  Iteration 6 (0.54
s, search 0.27 s): objective=8  Iteration 7 (0.61 s, search 0.31 s):
objective=842149 imbalance=1.872 nsplit=2 Converged at iteration 7: objective
did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 1000000    | 4096       | **SIVF**        | 0.2305     | 4338943         | - |
| "          | "          | Vanilla         | 0.6029     | 1658633 | 2.62      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points Clustering 262144 points in 128D to 16384 clusters, redo
1 times, 20 iterations Preprocessing in 0.14 s Iteration 0 (3.23 s, search 0.16
s): objective=6  Iteration 1 (6.45 s, search 0.32 s): objective=3  Iteration 2
(9.62 s, search 0.49 s): objective=1  Iteration 3 (12.83 s, search 0.66 s):
objective=  Iteration 6 (22.61 s, search 1.21 s): objective=12106
imbalance=1.656 nsplit=6434   Iteration 7 (25.86 s, search 1.38 s):
objective=9895.32 imbalance=1.650 nsplit=641  Iteration 8 (29.09 s, search 1.55
s): objective=6604.51 imbalance=1.648 nsplit=641  Iteration 9 (32.31 s,
search 1.72 s): objective=5594.58 imbalance=1.647 nsplit=640  Iteration 10
(35.55 s, search 1.89 s): objective=4992.18 imbalance=1.645 nsplit=64  Iteration
11 (38.77 s, search 2.05 s): objective=3820.45 imbalance=1.644 nsplit=64
Iteration 12 (42.00 s, search 2.22 s): objective=3326.08 imbalance=1.644
nsplit=63  Iteration 13 (45.23 s, search 2.39 s): objective=2918.26
imbalance=1.643 nsplit=63  Iteration 14 (48.53 s, search 2.56 s):
objective=2740.02 imbalance=1.642 nsplit=63  Iteration 15 (51.83 s, search 2.72
s): objective=2155.25 imbalance=1.642 nsplit=63  Iteration 16 (55.07 s,
search 2.89 s): objective=1772.41 imbalance=1.641 nsplit=63  Iteration 17 (58.32
s, search 3.06 s): objective=1587.08 imbalance=1.641 nsplit=63  Iteration 18
(61.57 s, search 3.22 s): objective=1413.33 imbalance=1.641 nsplit=63  Iteration
19 (64.87 s, search 3.39 s): objective=1413.36 imbalance=1.641 nsplit=6390
[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 1000000    | 16384      | **SIVF**        | 0.7511     | 1331430         | - |
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points | "          | "          | Vanilla         | 1.5625 |
640010          | 2.08      x |
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 0 (0.02 s, search 0.01 s): objective=886926 imbalance=2.415
nsplit=52     Iteration 1 (0.04 s, search 0.02 s): objective=562100
imbalance=1.992 nsplit=2      Iteration 2 (0.05 s, search 0.03 s):
objective=555369 imbalance=1.949 nsplit=0      Iteration 3 (0.07 s, search 0.04
s): objective=554004 imbalance=1.930 nsplit=0      Iteration 4 (0.08 s, search
0.05 s): objective=553472 imbalance=1.925 nsplit=0      Iteration 5 (0.10 s,
search 0.06 s): objective=553294 imbalance=1.924 nsplit=0      Iteration 6 (0.11
s, search 0.07 s): objective=553246 imbalance=1.923 nsplit=0      Iteration 7
(0.13 s, search 0.08 s): objective=553199 imbalance=1.923 nsplit=0 Iteration 8
(0.14 s, search 0.09 s): objective=553151 imbalance=1.923 nsplit=0 Iteration 9
(0.16 s, search 0.10 s): objective=553140 imbalance=1.923 nsplit=0 Iteration 10
(0.17 s, search 0.12 s): objective=553137 imbalance=1.923 nsplit=0     Iteration
11 (0.19 s, search 0.13 s): objective=553137 imbalance=1.923 nsplit=0 Converged
at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 5000000    | 1024       | **SIVF**        | 1.4787     | 3381303         | - |
| "          | "          | Vanilla         | 2.2546     | 2217715 | 1.52      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.08 s
  Iteration 0 (0.14 s, search 0.04 s): objective=1.55679e+06 imbalance=3.080
nsplit=  Iteration 1 (0.23 s, search 0.08 s): objective=946911 imbalance=2.104
nsplit=217    Iteration 2 (0.29 s, search 0.11 s): objective=873374
imbalance=1.938 nsplit=74     Iteration 3 (0.36 s, search 0.15 s):
objective=852494 imbalance=1.885 nsplit=20     Iteration 4 (0.41 s, search 0.19
s): objective=844697 imbalance=1.873 nsplit=5      Iteration 5 (0.47 s, search
0.23 s): objective=842405 imbalance=1.873 nsplit=3      Iteration 6 (0.54 s,
search 0.27 s): objective=842149 imbalance=1.873 nsplit=3      Iteration 7 (0.61
s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2 Converged at
iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 5000000    | 4096       | **SIVF**        | 1.2384     | 4037435         | - |
| "          | "          | Vanilla         | 2.8003     | 1785504 | 2.26      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points Clustering 262144 points in 128D to 16384 clusters, redo
1 times, 20 iterations Preprocessing in 0.11 s Iteration 0 (3.22 s, search 0.16
s): objective=687716 imbalance=2.538 nsplit=8259   Iteration 1 (6.41 s, search
0.32 s): objective=370994 imbalance=1.918 nsplit=7093   Iteration 2 (9.58 s,
search 0.49 s): objective=145721 imbalance=1.739 nsplit=6669   Iteration 3
(12.79 s, search 0.66 s): objective=58999.8 imbalance=1.688 nsplit=653 Iteration
4 (16.01 s, search 0.83 s): objective=29620.5 imbalance=1.667 nsplit=646
Iteration 5 (19.22 s, search 1.00 s): objective=16254 imbalance=1.660
nsplit=6445   Iteration 6 (22.45 s, search 1.16 s): objective=12106
imbalance=1.656 nsplit=6434   Iteration 7 (25.69 s, search 1.33 s):
objective=9895.32 imbalance=1.650 nsplit=641  Iteration 8 (28.90 s, search 1.50
s): objective=6604.51 imbalance=1.648 nsplit=641  Iteration 9 (32.12 s,
search 1.67 s): objective=5594.58 imbalance=1.647 nsplit=640  Iteration 10
(35.34 s, search 1.84 s): objective=4992.18 imbalance=1.645 nsplit=64  Iteration
11 (38.56 s, search 2.00 s): objective=3820.45 imbalance=1.644 nsplit=64
Iteration 12 (41.77 s, search 2.17 s): objective=3326.08 imbalance=1.644
nsplit=63  Iteration 13 (44.98 s, search 2.34 s): objective=2918.26
imbalance=1.643 nsplit=63  Iteration 14 (48.23 s, search 2.51 s):
objective=2740.02 imbalance=1.642 nsplit=63  Iteration 15 (51.43 s, search 2.68
s): objective=2155.25 imbalance=1.642 nsplit=63  Iteration 16 (54.63 s,
search 2.84 s): objective=1772.41 imbalance=1.641 nsplit=63  Iteration 17 (57.87
s, search 3.01 s): objective=1587.08 imbalance=1.641 nsplit=63  Iteration 18
(61.10 s, search 3.18 s): objective=1413.33 imbalance=1.641 nsplit=63  Iteration
19 (64.32 s, search 3.34 s): objective=1413.36 imbalance=1.641 nsplit=6390
[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 5000000    | 16384      | **SIVF**        | 3.4174     | 1463102         | - |
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points | "          | "          | Vanilla         | 7.0367 |
710564          | 2.06      x |
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 65536 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 0 (0.02 s, search 0.01 s): objective=886926 imbalance=2.415
nsplit=52     Iteration 1 (0.04 s, search 0.02 s): objective=562100
imbalance=1.992 nsplit=2      Iteration 2 (0.05 s, search 0.03 s):
objective=555369 imbalance=1.949 nsplit=0      Iteration 3 (0.07 s, search 0.04
s): objective=554004 imbalance=1.930 nsplit=0      Iteration 4 (0.08 s, search
0.05 s): objective=553472 imbalance=1.925 nsplit=0      Iteration 5 (0.10 s,
search 0.06 s): objective=553294 imbalance=1.924 nsplit=0      Iteration 6 (0.11
s, search 0.07 s): objective=553246 imbalance=1.923 nsplit=0      Iteration 7
(0.13 s, search 0.08 s): objective=553199 imbalance=1.923 nsplit=0 Iteration 8
(0.14 s, search 0.09 s): objective=553151 imbalance=1.923 nsplit=0 Iteration 9
(0.16 s, search 0.10 s): objective=553140 imbalance=1.923 nsplit=0 Iteration 10
(0.17 s, search 0.11 s): objective=553137 imbalance=1.923 nsplit=0     Iteration
11 (0.19 s, search 0.13 s): objective=553137 imbalance=1.923 nsplit=0 Converged
at iteration 11: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
| 10000000   | 1024       | **SIVF**        | 2.4028     | 4161750         | - |
| "          | "          | Vanilla         | 4.4297     | 2257512 | 1.84      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 163840 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.06 s
  Iteration 0 (0.14 s, search 0.04 s): objective=1.55679e+06 imbalance=3.080
nsplit=  Iteration 1 (0.23 s, search 0.08 s): objective=946911 imbalance=2.104
nsplit=217    Iteration 2 (0.29 s, search 0.11 s): objective=873374
imbalance=1.938 nsplit=74     Iteration 3 (0.36 s, search 0.15 s):
objective=852494 imbalance=1.885 nsplit=20     Iteration 4 (0.42 s, search 0.19
s): objective=844697 imbalance=1.873 nsplit=5      Iteration 5 (0.47 s, search
0.23 s): objective=842405 imbalance=1.873 nsplit=3      Iteration 6 (0.53 s,
search 0.27 s): objective=842149 imbalance=1.873 nsplit=3      Iteration 7 (0.59
s, search 0.31 s): objective=842149 imbalance=1.872 nsplit=2 Converged at
iteration 7: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
| 10000000   | 4096       | **SIVF**        | 2.4821     | 4028908         | - |
| "          | "          | Vanilla         | 5.4864     | 1822672 | 2.21      x
|
|------------|------------|-----------------|------------|-----------------|------------|
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points Clustering 262144 points in 128D to 16384 clusters, redo
1 times, 20 iterations Preprocessing in 0.10 s Iteration 0 (3.24 s, search 0.15
s): objective=687716 imbalance=2.538 nsplit=8259   Iteration 1 (6.47 s, search
0.32 s): objective=370994 imbalance=1.918 nsplit=7093   Iteration 2 (9.72 s,
search 0.49 s): objective=145721 imbalance=1.739 nsplit=6669   Iteration 3
(12.94 s, search 0.66 s): objective=58999.8 imbalance=1.688 nsplit=653 Iteration
4 (16.19 s, search 0.82 s): objective=29620.5 imbalance=1.667 nsplit=646
Iteration 5 (19.43 s, search 0.99 s): objective=16254 imbalance=1.660
nsplit=6445   Iteration 6 (22.74 s, search 1.16 s): objective=12106
imbalance=1.656 nsplit=6434   Iteration 7 (25.99 s, search 1.33 s):
objective=9895.32 imbalance=1.650 nsplit=641  Iteration 8 (29.23 s, search 1.50
s): objective=6604.51 imbalance=1.648 nsplit=641  Iteration 9 (32.45 s,
search 1.67 s): objective=5594.58 imbalance=1.647 nsplit=640  Iteration 10
(35.69 s, search 1.83 s): objective=4992.18 imbalance=1.645 nsplit=64  Iteration
11 (38.90 s, search 2.00 s): objective=3820.45 imbalance=1.644 nsplit=64
Iteration 12 (42.11 s, search 2.17 s): objective=3326.08 imbalance=1.644
nsplit=63  Iteration 13 (45.33 s, search 2.34 s): objective=2918.26
imbalance=1.643 nsplit=63  Iteration 14 (48.59 s, search 2.51 s):
objective=2740.02 imbalance=1.642 nsplit=63  Iteration 15 (51.80 s, search 2.67
s): objective=2155.25 imbalance=1.642 nsplit=63  Iteration 16 (55.00 s,
search 2.83 s): objective=1772.41 imbalance=1.641 nsplit=63  Iteration 17 (58.23
s, search 3.00 s): objective=1587.08 imbalance=1.641 nsplit=63  Iteration 18
(61.46 s, search 3.16 s): objective=1413.33 imbalance=1.641 nsplit=63  Iteration
19 (64.68 s, search 3.33 s): objective=1413.36 imbalance=1.641 nsplit=6390
[SIVF::train] GPU K-Means complete. Quantizer populated with 16384 centroids.
| 10000000   | 16384      | **SIVF**        | 6.0206     | 1660955         | - |
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points | "          | "          | Vanilla         | 14.7630 |
677367          | 2.45      x |
|------------|------------|-----------------|------------|-----------------|------------|
cc@rtx6000:~/ElasticIVF/build$
 */