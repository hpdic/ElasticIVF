/**
 * faiss/hpdic/experiment/test_sivf_search.cpp
 * Benchmark: ElasticIVF Search vs Vanilla Faiss IVFFlat
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

void generate_data(size_t n, int d, std::vector<float>& data) {
    for (size_t i = 0; i < n * d; ++i) {
        data[i] = (float)drand48();
    }
}

int main() {
    // ==========================================
    // 配置
    // ==========================================
    int d = 128;
    int nlist = 1024;
    size_t nb = 100000; // 数据库大小 (10万)
    size_t nq = 1000;   // 查询数量
    int k = 10;         // Top-K
    int nprobe = 10;    // 探测桶数 (通常 nlist 的 1%~5%)

    // SIVF 显存池配置
    size_t max_vectors = nb * 2;
    size_t slab_pool_size = nb * 2; // 200,000 Slabs (huge pool)
    
    printf("==================================================\n");
    printf(" BENCHMARK: Search Verify (SIVF vs Vanilla)\n");
    printf(" d=%d, nlist=%d, nb=%ld, nq=%ld, k=%d, nprobe=%d\n",
           d,
           nlist,
           nb,
           nq,
           k,
           nprobe);
    printf("==================================================\n\n");

    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024);

    GpuIndexIVFConfig config;
    config.device = 0;

    // 准备数据
    // 我们用数据库的前 nq 个向量作为 query，这样 Top-1 距离应该是 0
    // (self-search)
    std::vector<float> xb(nb * d);
    generate_data(nb, d, xb);

    // Query 就是 Database 的前部分
    std::vector<float> xq(nq * d);
    memcpy(xq.data(), xb.data(), nq * d * sizeof(float));

    std::vector<idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i)
        ids[i] = i;

    // 结果容器
    std::vector<float> dists(nq * k);
    std::vector<idx_t> labels(nq * k);

    // ==========================================
    // Round 1: ElasticIVF (SIVF)
    // ==========================================
    {
        printf("[ElasticIVF] Setting up...\n");
        GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
        index.initSlabManager(max_vectors, slab_pool_size);
        index.nprobe = nprobe; // 设置查询参数

        // [DEBUG] 打印训练前状态
        printf("DEBUG: Before train, is_trained = %s\n",
               index.is_trained ? "TRUE" : "FALSE");

        printf("[ElasticIVF] Training & Adding...\n");
        // 为了省事，直接用 xb 训练
        index.train(std::min(nb, (size_t)65536), xb.data());

        // [DEBUG] 打印训练后状态
        printf("DEBUG: After train, is_trained = %s\n",
               index.is_trained ? "TRUE" : "FALSE");
        printf("DEBUG: Quantizer ntotal = %ld (Should be %d)\n",
               index.quantizer->ntotal,
               nlist);

        index.add_with_ids(nb, xb.data(), ids.data());

        printf("[ElasticIVF] Searching...\n");
        cudaDeviceSynchronize();
        double t0 = elapsed();

        index.search(nq, xq.data(), k, dists.data(), labels.data());

        cudaDeviceSynchronize();
        double t1 = elapsed();

        printf("-> Search Time: %.4fs | QPS: %.2f\n",
               (t1 - t0),
               nq / (t1 - t0));

        // 验证正确性
        int match_count = 0;
        for (int i = 0; i < nq; ++i) {
            // Self-search, distance should be very close to 0
            if (dists[i * k] < 1e-4) {
                match_count++;
            }
        }
        printf("-> Accuracy Check (Recall@1 with Dist~0): %d / %ld (%.2f%%)\n\n",
               match_count,
               nq,
               100.0 * match_count / nq);

        // 打印前几个结果看看
        printf("   Top-3 results for Query 0:\n");
        for (int j = 0; j < 3; ++j) {
            printf("   Rank %d: ID=%ld, Dist=%.5f\n", j, labels[j], dists[j]);
        }
        printf("\n");
    }

    // ==========================================
    // Round 2: Vanilla Faiss
    // ==========================================
    {
        printf("[Vanilla Faiss] Setting up...\n");
        GpuIndexIVFFlatConfig flatConfig;
        flatConfig.device = 0;
        faiss::gpu::GpuIndexIVFFlat index(
                &res, d, nlist, METRIC_L2, flatConfig);
        index.nprobe = nprobe;

        printf("[Vanilla Faiss] Training & Adding...\n");
        index.train(std::min(nb, (size_t)65536), xb.data());

        index.add_with_ids(nb, xb.data(), ids.data());

        printf("[Vanilla Faiss] Searching...\n");
        cudaDeviceSynchronize();
        double t0 = elapsed();

        index.search(nq, xq.data(), k, dists.data(), labels.data());

        cudaDeviceSynchronize();
        double t1 = elapsed();

        printf("-> Search Time: %.4fs | QPS: %.2f\n",
               (t1 - t0),
               nq / (t1 - t0));

        // 验证
        int match_count = 0;
        for (int i = 0; i < nq; ++i) {
            if (dists[i * k] < 1e-4)
                match_count++;
        }
        printf("-> Accuracy Check: %d / %ld (%.2f%%)\n",
               match_count,
               nq,
               100.0 * match_count / nq);
        printf("\n");
    }

    return 0;
}