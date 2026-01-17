/**
 * test_sivf_nonivf.cpp
 * 
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 * 
 * * Extended Landscape Analysis:
 * - GPU Flat (Baseline)
 * - CPU HNSW (Graph Baseline)
 * - CPU LSH (Hash Baseline)  
 * - CPU NSG (Graph Baseline) 
 * - SIVF (Ours)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <sys/stat.h>
#include <omp.h>

// Faiss Headers
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexLSH.h>    
#include <faiss/IndexNSG.h>    
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h> 
#include <faiss/gpu/GpuIndexSIVF.h>

using namespace faiss::gpu;

// --- Loaders (Same as before) ---
inline bool file_exists(const char* name) { struct stat buffer; return (stat(name, &buffer) == 0); }
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r"); if (!f) exit(1);
    int d; fread(&d, 1, sizeof(int), f); *d_out = (size_t)d;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    *n_out = size / (sizeof(int) + d * sizeof(float));
    float* x = new float[*n_out * *d_out];
    size_t nr = 0;
    for (size_t i = 0; i < *n_out; i++) {
        int d_check; fread(&d_check, 1, sizeof(int), f);
        nr += fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f); return x;
}
float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb"); if (!f) exit(1);
    int n_in, d_in; fread(&n_in, sizeof(int), 1, f); fread(&d_in, sizeof(int), 1, f);
    *n_out = (size_t)n_in; *d_out = (size_t)d_in;
    float* data = new float[*n_out * *d_out];
    fread(data, sizeof(float), *n_out * *d_out, f); fclose(f); return data;
}
float* load_any(std::string path, size_t* d, size_t* n) {
    if (path.find(".fvecs") != std::string::npos) return fvecs_read(path.c_str(), d, n);
    else return fbin_read(path.c_str(), d, n);
}

template<typename Func>
double measure_ms(Func f) {
    auto t1 = std::chrono::high_resolution_clock::now();
    f(); cudaDeviceSynchronize(); 
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

void run_dataset(std::string name, std::string path, StandardGpuResources& res, size_t limit_n = 1000000) {
    size_t d, nb;
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << " Dataset: " << name << " (" << path << ")" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    
    float* xb = load_any(path, &d, &nb);
    if(limit_n > nb) limit_n = nb;
    std::cout << "  -> Loaded Limit: N=" << limit_n << ", D=" << d << std::endl;

    size_t del_bs = 10000;
    if (del_bs > limit_n) del_bs = limit_n / 10;
    std::vector<faiss::idx_t> del_ids(del_bs);
    std::iota(del_ids.begin(), del_ids.end(), 0);
    faiss::IDSelectorBatch sel(del_bs, del_ids.data());

    // --- 1. GPU Flat ---
    {
        std::cout << "  [1] GPU Flat" << std::endl;
        faiss::gpu::GpuIndexFlatConfig config; config.device = 0;
        faiss::gpu::GpuIndexFlatL2 index(&res, d, config);

        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(limit_n, xb);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(limit_n/sec) << " vec/s" << std::endl;
        
        double ms = measure_ms([&](){
            faiss::Index* cpu = faiss::gpu::index_gpu_to_cpu(&index);
            cpu->remove_ids(sel);
            auto gpu = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu);
            delete cpu; delete gpu;
        });
        std::cout << "      Del: " << ms << " ms (Roundtrip)" << std::endl;
    }

    // --- 2. CPU HNSW ---
    {
        std::cout << "  [2] CPU HNSW (M=32)" << std::endl;
        faiss::IndexHNSWFlat index(d, 32);
        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(limit_n, xb);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(limit_n/sec) << " vec/s" << std::endl;
        try { index.remove_ids(sel); std::cout << "      Del: Success" << std::endl; } 
        catch (...) { std::cout << "      Del: N/A (Not Supported)" << std::endl; }
    }

    // --- 3. CPU LSH (New) ---
    {
        // nbits = d * 2 is a common heuristic for reasonable recall
        int nbits = (d < 64) ? 128 : d * 2; 
        std::cout << "  [3] CPU LSH (nbits=" << nbits << ")" << std::endl;
        faiss::IndexLSH index(d, nbits);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(limit_n, xb);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(limit_n/sec) << " vec/s" << std::endl;

        try { 
            auto d1 = std::chrono::high_resolution_clock::now();
            index.remove_ids(sel); 
            auto d2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(d2 - d1).count();
            std::cout << "      Del: " << ms << " ms" << std::endl; 
        } 
        catch (...) { std::cout << "      Del: N/A (Not Supported)" << std::endl; }
    }

    // --- 4. CPU NSG (New) ---
    {
        // R=32 is comparable to HNSW M=32
        std::cout << "  [4] CPU NSG (R=32)" << std::endl;
        faiss::IndexNSGFlat index(d, 32, faiss::METRIC_L2);
        
        // NSG typically requires training/building graph, unlike HNSW dynamic add
        // But IndexNSGFlat supports add() which triggers build/search
        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(limit_n, xb);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(limit_n/sec) << " vec/s" << std::endl;

        try { index.remove_ids(sel); std::cout << "      Del: Success" << std::endl; } 
        catch (...) { std::cout << "      Del: N/A (Not Supported)" << std::endl; }
    }

    // --- 5. SIVF (Ours) ---
    {
        std::cout << "  [5] SIVF" << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, 1024, faiss::METRIC_L2, config);
        
        size_t cap_alloc = (d > 512) ? limit_n : (size_t)(limit_n * 1.2);
        index.initSlabManager(cap_alloc, d);
        
        size_t n_train = std::min((size_t)50000, limit_n);
        index.train(n_train, xb); 
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(limit_n, xb);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(limit_n/sec) << " vec/s" << std::endl;

        double ms = measure_ms([&](){ index.remove_ids(sel); });
        std::cout << "      Del: " << ms << " ms (Native)" << std::endl;
    }

    delete[] xb;
}

int main() {
    omp_set_num_threads(48); // Use all CPU cores
    StandardGpuResources res;
    res.setTempMemory(1024L * 1024 * 1024);

    std::string root = "/home/cc/ElasticIVF/hpdic/data/";
    
    // 1. SIFT (128D)
    run_dataset("SIFT1M", root + "sift/sift_base.fvecs", res, 1000000);

    // 2. T2I (200D)
    run_dataset("T2I-1M", root + "t2i/t2i_base_1M.fbin", res, 1000000);

    // 3. GIST (960D) - Limit 200k
    // NSG on GIST will be extremely slow, 200k is necessary
    run_dataset("GIST1M", root + "gist/gist_base.fvecs", res, 200000);

    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_nonivf

----------------------------------------------------------
 Dataset: SIFT1M (/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs)
----------------------------------------------------------
  -> Loaded Limit: N=1000000, D=128
  [1] GPU Flat
[HPDIC MOD] Faiss GPU initialized on device ID: 0
      Add: 9169323 vec/s
      Del: 835.772 ms (Roundtrip)
  [2] CPU HNSW (M=32)
      Add: 25505 vec/s
      Del: N/A (Not Supported)
  [3] CPU LSH (nbits=256)
      Add: 787134 vec/s
      Del: 14.6407 ms
  [4] CPU NSG (R=32)
      Add: 3712 vec/s
      Del: N/A (Not Supported)
  [5] SIVF

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 191596
  > Data Buffer: 1200000 -> 6131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 19 (0.25 s, search 0.18 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
      Add: 1714218 vec/s
      Del: 0.936983 ms (Native)

----------------------------------------------------------
 Dataset: T2I-1M (/home/cc/ElasticIVF/hpdic/data/t2i/t2i_base_1M.fbin)
----------------------------------------------------------
  -> Loaded Limit: N=1000000, D=200
  [1] GPU Flat
      Add: 6020397 vec/s
      Del: 1160.55 ms (Roundtrip)
  [2] CPU HNSW (M=32)
      Add: 25733 vec/s
      Del: N/A (Not Supported)
  [3] CPU LSH (nbits=400)
      Add: 427558 vec/s
      Del: 16.3462 ms
  [4] CPU NSG (R=32)
      Add: 3179 vec/s
      Del: N/A (Not Supported)
  [5] SIVF

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200 -> 191596
  > Data Buffer: 1200000 -> 6131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 200D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.04 s
  Iteration 19 (0.45 s, search 0.25 s): objective=22642.1 imbalance=1.206 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
      Add: 1853782 vec/s
      Del: 1.54933 ms (Native)

----------------------------------------------------------
 Dataset: GIST1M (/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs)
----------------------------------------------------------
  -> Loaded Limit: N=200000, D=960
  [1] GPU Flat
      Add: 1230089 vec/s
      Del: 1143.16 ms (Roundtrip)
  [2] CPU HNSW (M=32)
      Add: 561 vec/s
      Del: N/A (Not Supported)
  [3] CPU LSH (nbits=1920)
      Add: 36602 vec/s
      Del: 9.17549 ms
  [4] CPU NSG (R=32)
      Add: 680 vec/s
      Del: N/A (Not Supported)
  [5] SIVF

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   960 -> 35346
  > Data Buffer: 200000 -> 1131072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 960D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.23 s
  Iteration 19 (1.48 s, search 1.02 s): objective=53878.4 imbalance=1.762 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
      Add: 427728 vec/s
      Del: 1.39925 ms (Native)
cc@rtx6000:~/ElasticIVF/build$ 
 */