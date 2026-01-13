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