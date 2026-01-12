/**
 * faiss/hpdic/experiment/test_sivf_add.cpp
 * * Benchmark for GpuIndexSIVF add functionality
 */

#include <sys/time.h>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

using namespace faiss;
using namespace faiss::gpu;

// 简单的计时器
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 随机数据生成器
void generate_data(size_t n, int d, std::vector<float>& data) {
    for (size_t i = 0; i < n * d; ++i) {
        data[i] = (float)drand48();
    }
}

int main() {
    // 1. 参数设置
    int d = 128;        // 维度
    int nlist = 1024;   // 聚类中心数 (Slab 链表数)
    size_t nt = 10000;  // 训练数据量
    size_t nb = 100000; // 插入数据库大小 (100k)

    // 预估最大容量 (用于 AddressTable)
    size_t max_vectors = nb * 2;
    // Slab池子大小 (稍微给大一点，避免碎片导致申请不到)
    size_t slab_pool_size = max_vectors / 32 + 10000;

    printf("Testing GpuIndexSIVF Add...\n");
    printf("d=%d, nlist=%d, nb=%ld\n", d, nlist, nb);

    // 2. 初始化 GPU 资源
    StandardGpuResources res;
    // 显存预分配设为临时区大小，避免和 SlabManager 抢显存
    res.setTempMemory(256 * 1024 * 1024);

    GpuIndexIVFConfig config;
    config.device = 0; // 使用 GPU 0

    // 3. 创建索引
    GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);

    // 4. 初始化内存引擎 (这一步非常关键)
    printf("Initializing SlabManager...\n");
    index.initSlabManager(max_vectors, slab_pool_size);

    // 5. 训练 (Train)
    // SIVF 依赖 Quantizer 来决定向量去哪个链表，所以必须先 Train
    printf("Generating training data (%ld vectors)...\n", nt);
    std::vector<float> xt(nt * d);
    generate_data(nt, d, xt);

    printf("Training...\n");
    index.train(nt, xt.data());
    printf("Training done.\n");

    // 6. 插入 (Add) - 这是我们今天测试的主角
    printf("Generating database data (%ld vectors)...\n", nb);
    std::vector<float> xb(nb * d);
    generate_data(nb, d, xb);

    // 生成顺序 ID
    std::vector<idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i) {
        ids[i] = i;
    }

    printf("Adding vectors to SIVF...\n");
    double t0 = elapsed();

    // 调用 add_with_ids (它内部会调用我们写的 addImpl_)
    index.add_with_ids(nb, xb.data(), ids.data());

    // 这里的同步是为了计时准确，因为 Kernel 是异步的
    cudaDeviceSynchronize();
    double t1 = elapsed();

    printf("===========================================\n");
    printf("Success! Added %ld vectors.\n", nb);
    printf("Time elapsed: %.4f s\n", t1 - t0);
    printf("Throughput:   %.2f vectors/sec\n", nb / (t1 - t0));
    printf("===========================================\n");

    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_add
Testing GpuIndexSIVF Add...
d=128, nlist=1024, nb=100000
[HPDIC MOD] Faiss GPU initialized on device ID: 0
Initializing SlabManager...
Generating training data (10000 vectors)...
Training...
Training done.
Generating database data (100000 vectors)...
Adding vectors to SIVF...
===========================================
Success! Added 100000 vectors.
Time elapsed: 0.0128 s
Throughput:   7788720.73 vectors/sec
===========================================
cc@rtx6000:~/ElasticIVF/build$
 */