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
 * Example
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_add
Preparing Data (Max NB=10000000, Max NT=262144)...

| NB         | nlist      | System          | Time(s)    | QPS (vec/s)     |
Speedup    |
|------------|------------|-----------------|------------|-----------------|------------|
[HPDIC MOD] Faiss GPU initialized on device ID: 0
| 1000000    | 1024       | **SIVF**        | 0.1126     | 8884938         | - |
| "          | "          | Vanilla         | 0.4786     | 2089515 | 4.25      x
|
|------------|------------|-----------------|------------|-----------------|------------|
| 1000000    | 4096       | **SIVF**        | 0.1120     | 8925076         | - |
| "          | "          | Vanilla         | 0.5985     | 1670939 | 5.34      x
|
|------------|------------|-----------------|------------|-----------------|------------|
| 1000000    | 16384      | **SIVF**        | 0.1115     | 8972541         | - |
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points | "          | "          | Vanilla         | 1.5848 |
631012          | 14.22     x |
|------------|------------|-----------------|------------|-----------------|------------|
| 5000000    | 1024       | **SIVF**        | 1.2218     | 4092307         | - |
| "          | "          | Vanilla         | 2.1938     | 2279141 | 1.80      x
|
|------------|------------|-----------------|------------|-----------------|------------|
| 5000000    | 4096       | **SIVF**        | 0.5539     | 9027568         | - |
| "          | "          | Vanilla         | 2.7217     | 1837076 | 4.91      x
|
|------------|------------|-----------------|------------|-----------------|------------|
| 5000000    | 16384      | **SIVF**        | 0.5526     | 9048614         | - |
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points | "          | "          | Vanilla         | 6.3589 |
786301          | 11.51     x |
|------------|------------|-----------------|------------|-----------------|------------|
| 10000000   | 1024       | **SIVF**        | 1.9795     | 5051730         | - |
| "          | "          | Vanilla         | 4.2533     | 2351118 | 2.15      x
|
|------------|------------|-----------------|------------|-----------------|------------|
| 10000000   | 4096       | **SIVF**        | 1.1170     | 8952879         | - |
| "          | "          | Vanilla         | 5.3545     | 1867595 | 4.79      x
|
|------------|------------|-----------------|------------|-----------------|------------|
| 10000000   | 16384      | **SIVF**        | 1.2887     | 7759836         | - |
WARNING clustering 262144 points to 16384 centroids: please provide at least
638976 training points | "          | "          | Vanilla         | 14.4242 |
693278          | 11.19     x |
|------------|------------|-----------------|------------|-----------------|------------|
 */