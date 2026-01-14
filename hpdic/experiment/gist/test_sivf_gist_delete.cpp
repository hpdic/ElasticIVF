#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// 头文件补全
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexSIVF.h> 
#include "gist_loader.h"

using namespace faiss::gpu;

int main() {
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
    size_t nb = 1000000; 
    int nlist = 1024;
    size_t n_delete = 10000; 

    size_t d, fnb;
    // 这里其实不需要重新读文件来获取 d，不过为了保险还是读一下头
    // 也可以硬编码 d=960, fnb=1000000 以节省时间，但读一下头很快
    float* xb = fvecs_read(base_file, &d, &fnb);
    if(nb > fnb) nb = fnb;

    // 随机删除 ID
    std::vector<faiss::idx_t> ids(nb);
    for(size_t i=0; i<nb; ++i) ids[i] = i;
    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    ids.resize(n_delete);
    faiss::IDSelectorBatch sel(n_delete, ids.data());

    // 调小 Temp，给 GIST 数据腾地方
    StandardGpuResources res;
    res.setTempMemory(512 * 1024 * 1024); 
    faiss::IndexFlatL2 quantizer(d);

    // ==========================================
    // Round 1: Baseline (Roundtrip)
    // ==========================================
    {
        std::cout << "[Baseline] Preparing..." << std::endl;
        faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2);
        index.train(50000, xb); // 训练少一点省时间
        index.add(nb, xb);
        cudaDeviceSynchronize();

        std::cout << "[Baseline] Deleting (Roundtrip 3.8GB data!)..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // 这三步是罪魁祸首
        faiss::Index* cpu = faiss::gpu::index_gpu_to_cpu(&index); 
        cpu->remove_ids(sel); 
        faiss::gpu::GpuIndexIVFFlat* new_gpu = 
            dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(faiss::gpu::index_cpu_to_gpu(&res, 0, cpu)); 
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "[Baseline] Time: " << ms << " ms" << std::endl;
        
        delete cpu; delete new_gpu;
    } 

    // 强制同步清理显存
    cudaDeviceSynchronize();

    // ==========================================
    // Round 2: SIVF
    // ==========================================
    {
        std::cout << "[SIVF] Preparing..." << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig cfg; cfg.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, cfg);
        
        // 关键修复：去掉 1.5 倍余量，只申请 1.0 (exact capacity)
        size_t cap = nb; 
        std::cout << "[SIVF] Allocating exact capacity: " << cap << std::endl;
        index.initSlabManager(cap, d);

        index.train(50000, xb);
        index.add(nb, xb);
        cudaDeviceSynchronize();

        std::cout << "[SIVF] Deleting (Native)..." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // 原生删除，无需搬运
        index.remove_ids(sel);
        
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "[SIVF] Time: " << ms << " ms" << std::endl;
    }

    delete[] xb;
    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ make test_sivf_gist_delete -j
./test_sivf_gist_delete
[ 65%] Built target faiss_gpu_objs
[100%] Built target faiss
[100%] Building CXX object CMakeFiles/test_sivf_gist_delete.dir/hpdic/experiment/gist/test_sivf_gist_delete.cpp.o
[100%] Linking CXX executable test_sivf_gist_delete
[100%] Built target test_sivf_gist_delete
[Baseline] Preparing...
[HPDIC MOD] Faiss GPU initialized on device ID: 0
[Baseline] Deleting (Roundtrip 3.8GB data!)...
[Baseline] Time: 11842.9 ms
[SIVF] Preparing...
[SIVF] Allocating exact capacity: 1000000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 160346
  > Data Buffer: 1000000 -> 5131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.23 s
  Iteration 19 (1.51 s, search 1.01 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
[SIVF] Deleting (Native)...
[SIVF] Time: 0.88981 ms
cc@rtx6000:~/ElasticIVF/build$ 
 */