#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/IDSelector.h>

// 辅助函数：生成随机向量
void generate_data(size_t n, int d, std::vector<float>& data) {
    for (size_t i = 0; i < n * d; ++i) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main() {
    // ==========================================
    // 核心参数配置
    // ==========================================
    int d = 128;
    int nlist = 4096;
    size_t nb = 1000000; // 1M 底库

    // [修改] 这里改成 10000，跟你的 Baseline 对齐
    size_t n_delete = 10000;

    // Baseline 数据 (用于直接计算加速比)
    double baseline_time_ms = 202.2;

    // ==========================================

    // 1. 准备数据
    std::vector<float> xb(nb * d);
    std::cout << "Generating " << nb << " vectors..." << std::endl;
    generate_data(nb, d, xb);

    std::vector<faiss::idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i)
        ids[i] = i;

    // 2. 初始化 SIVF
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2);

    // 预分配显存池
    size_t max_vecs = (size_t)(nb * 1.5);
    index.initSlabManager(max_vecs, max_vecs / 32 + 20000);

    // 3. 训练与插入
    std::cout << "Training..." << std::endl;
    index.train(50000, xb.data());

    std::cout << "Adding " << nb << " vectors..." << std::endl;
    index.add_with_ids(nb, xb.data(), ids.data());

    cudaDeviceSynchronize();

    // 4. 准备删除列表 (删除前 10000 个)
    std::vector<faiss::idx_t> ids_to_delete;
    ids_to_delete.reserve(n_delete);
    for (size_t i = 0; i < n_delete; ++i) {
        ids_to_delete.push_back(ids[i]);
    }

    faiss::IDSelectorBatch selector(n_delete, ids_to_delete.data());

    // 5. Benchmark: 执行删除
    std::cout << "Benchmark: Removing " << n_delete << " vectors..."
              << std::endl;

    // 预热 GPU (可选)
    // cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    // 核心调用
    size_t removed_count = index.remove_ids(selector);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 6. 打印结果对比
    std::cout << "\n================ PERFORMANCE REPORT ================"
              << std::endl;
    std::cout << "Workload:        Delete " << n_delete << " vectors"
              << std::endl;
    std::cout << "Baseline (Faiss):" << baseline_time_ms << " ms" << std::endl;
    std::cout << "SIVF (Ours):     " << ms << " ms" << std::endl;
    std::cout << "----------------------------------------------------"
              << std::endl;
    std::cout << "Speedup:         " << baseline_time_ms / ms << " x"
              << std::endl;
    std::cout << "Latency per ID:  " << (ms / n_delete) * 1000 << " ns"
              << std::endl;
    std::cout << "Actual Removed:  " << removed_count
              << " (Recall Rate: " << (double)removed_count / n_delete * 100.0
              << "%)" << std::endl;
    std::cout << "====================================================\n"
              << std::endl;

    return 0;
}

/**
 * Example Output:
cc@rtx6000:~/ElasticIVF/build$ make test_sivf_delete -j
[ 65%] Built target faiss_gpu_objs
[100%] Built target faiss
[100%] Building CXX object faiss/gpu/CMakeFiles/test_sivf_delete.dir/__/__/hpdic/experiment/test_sivf_delete.cpp.o
[100%] Linking CXX executable test_sivf_delete
[100%] Built target test_sivf_delete
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_delete 
Generating 1000000 vectors...
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66875 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 50000 points to 4096 centroids: please provide at least 159744 training points
Clustering 50000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.39 s, search 0.26 s): objective=420117 imbalance=1.874 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Adding 1000000 vectors...
Benchmark: Removing 10000 vectors...

================ PERFORMANCE REPORT ================
Workload:        Delete 10000 vectors
Baseline (Faiss):202.2 ms
SIVF (Ours):     0.681474 ms
----------------------------------------------------
Speedup:         296.71 x
Latency per ID:  0.0681474 ns
Actual Removed:  10000 (Recall Rate: 100%)
====================================================

cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_delete 
Generating 1000000 vectors...
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66875 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 50000 points to 4096 centroids: please provide at least 159744 training points
Clustering 50000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.38 s, search 0.27 s): objective=420117 imbalance=1.874 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Adding 1000000 vectors...
Benchmark: Removing 10000 vectors...

================ PERFORMANCE REPORT ================
Workload:        Delete 10000 vectors
Baseline (Faiss):202.2 ms
SIVF (Ours):     0.683111 ms
----------------------------------------------------
Speedup:         295.999 x
Latency per ID:  0.0683111 ns
Actual Removed:  10000 (Recall Rate: 100%)
====================================================

cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ 
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_delete 
Generating 1000000 vectors...
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   66875 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

Training...
[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 50000 points to 4096 centroids: please provide at least 159744 training points
Clustering 50000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.39 s, search 0.27 s): objective=420117 imbalance=1.874 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
Adding 1000000 vectors...
Benchmark: Removing 10000 vectors...

================ PERFORMANCE REPORT ================
Workload:        Delete 10000 vectors
Baseline (Faiss):202.2 ms
SIVF (Ours):     0.66785 ms
----------------------------------------------------
Speedup:         302.763 x
Latency per ID:  0.066785 ns
Actual Removed:  10000 (Recall Rate: 100%)
====================================================

cc@rtx6000:~/ElasticIVF/build$ 
 */