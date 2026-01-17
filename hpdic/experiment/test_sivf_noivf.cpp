/**
 * benchmark_landscape.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 * 
 * The Ultimate "Landscape Analysis" Benchmark.
 * Compares SIVF against GPU Flat (Brute Force) and CPU HNSW (Graph)
 * across SIFT1M, T2I-1M, and GIST1M.
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
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h> 
#include <faiss/gpu/GpuIndexSIVF.h>

using namespace faiss::gpu;

// =========================================================
// Unified Loader Logic (Embedded to avoid header dependency hell)
// =========================================================
inline bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

// .fvecs loader (SIFT/GIST)
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Err: %s not found\n", fname); exit(1); }
    int d;
    fread(&d, 1, sizeof(int), f);
    *d_out = (size_t)d;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n_out = size / (sizeof(int) + d * sizeof(float));
    float* x = new float[*n_out * *d_out];
    size_t nr = 0;
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) exit(1);
        nr += fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f);
    return x;
}

// .fbin loader (T2I)
float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Err: %s not found\n", fname); exit(1); }
    int n_in, d_in;
    fread(&n_in, sizeof(int), 1, f);
    fread(&d_in, sizeof(int), 1, f);
    *n_out = (size_t)n_in;
    *d_out = (size_t)d_in;
    float* data = new float[*n_out * *d_out];
    fread(data, sizeof(float), *n_out * *d_out, f);
    fclose(f);
    return data;
}

// Universal Load
float* load_any(std::string path, size_t* d, size_t* n) {
    if (path.find(".fvecs") != std::string::npos) {
        return fvecs_read(path.c_str(), d, n);
    } else {
        return fbin_read(path.c_str(), d, n);
    }
}

// =========================================================
// Benchmarking Logic
// =========================================================
template<typename Func>
double measure_ms(Func f) {
    auto t1 = std::chrono::high_resolution_clock::now();
    f();
    cudaDeviceSynchronize(); 
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

void run_dataset(std::string name, std::string path, StandardGpuResources& res) {
    size_t d, nb;
    size_t nb_load = 1000000; // Limit to 1M
    
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << " Dataset: " << name << " (" << path << ")" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    
    float* xb = load_any(path, &d, &nb);
    if(nb_load > nb) nb_load = nb;
    std::cout << "  -> Loaded: N=" << nb_load << ", D=" << d << std::endl;

    // Delete Batch Setup
    size_t del_bs = 10000;
    std::vector<faiss::idx_t> del_ids(del_bs);
    std::iota(del_ids.begin(), del_ids.end(), 0);
    faiss::IDSelectorBatch sel(del_bs, del_ids.data());

    // --- 1. GPU Flat (Baseline: Max Throughput) ---
    {
        std::cout << "  [1] GPU Flat (GpuIndexFlatL2)" << std::endl;
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexFlatL2 index(&res, d, config);

        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(nb_load, xb);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(nb_load/sec) << " vec/s" << std::endl;

        // Simulate Roundtrip for Delete
        double ms = measure_ms([&](){
            faiss::Index* cpu = faiss::gpu::index_gpu_to_cpu(&index);
            cpu->remove_ids(sel);
            auto gpu = faiss::gpu::index_cpu_to_gpu(&res, 0, cpu);
            delete cpu; delete gpu;
        });
        std::cout << "      Del: " << ms << " ms (Roundtrip)" << std::endl;
    }

    // --- 2. CPU HNSW (Baseline: Dynamic Graph) ---
    {
        std::cout << "  [2] CPU HNSW (IndexHNSWFlat, M=32)" << std::endl;
        faiss::IndexHNSWFlat index(d, 32);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(nb_load, xb);
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(nb_load/sec) << " vec/s" << std::endl;

        // Try Delete (Expect Failure)
        try {
            index.remove_ids(sel); // This will throw
            std::cout << "      Del: Success (Unexpected!)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "      Del: N/A (Not Supported by HNSW)" << std::endl;
        }
    }

    // --- 3. SIVF (Ours) ---
    {
        std::cout << "  [3] SIVF (GpuIndexSIVF)" << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexSIVF index(&res, d, 1024, faiss::METRIC_L2, config);
        
        index.initSlabManager(nb_load * 1.5, d);
        index.train(50000, xb); 
        cudaDeviceSynchronize();

        auto t1 = std::chrono::high_resolution_clock::now();
        index.add(nb_load, xb);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "      Add: " << (size_t)(nb_load/sec) << " vec/s" << std::endl;

        double ms = measure_ms([&](){
            index.remove_ids(sel);
        });
        std::cout << "      Del: " << ms << " ms (Native)" << std::endl;
    }

    delete[] xb;
}

int main() {
    omp_set_num_threads(48); // Maximize CPU Power
    StandardGpuResources res;
    res.setTempMemory(1024L * 1024 * 1024); // 1GB

    // Dataset Registry
    std::string root = "/home/cc/ElasticIVF/hpdic/data/";
    
    // 1. SIFT (128D)
    run_dataset("SIFT1M", root + "sift/sift_base.fvecs", res);

    // 2. T2I (200D)
    run_dataset("T2I-1M", root + "t2i/t2i_base_1M.fbin", res);

    // 3. GIST (960D)
    run_dataset("GIST1M", root + "gist/gist_base.fvecs", res);

    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ make -j test_sivf_nonivf
[ 64%] Built target faiss_gpu_objs
[100%] Built target faiss
[100%] Building CXX object CMakeFiles/test_sivf_nonivf.dir/hpdic/experiment/test_sivf_noivf.cpp.o
[100%] Linking CXX executable test_sivf_nonivf
[100%] Built target test_sivf_nonivf
cc@rtx6000:~/ElasticIVF/build$ ./test_sivf_nonivf

----------------------------------------------------------
 Dataset: SIFT1M (/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs)
----------------------------------------------------------
  -> Loaded: N=1000000, D=128
  [1] GPU Flat (GpuIndexFlatL2)
[HPDIC MOD] Faiss GPU initialized on device ID: 0
      Add: 9323285 vec/s
      Del: 838.014 ms (Roundtrip)
  [2] CPU HNSW (IndexHNSWFlat, M=32)
      Add: 25555 vec/s
      Del: N/A (Not Supported by HNSW)
  [3] SIVF (GpuIndexSIVF)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   128 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 128D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.02 s
  Iteration 19 (0.32 s, search 0.19 s): objective=2.42526e+09 imbalance=1.242 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
      Add: 4894883 vec/s
      Del: 0.873553 ms (Native)

----------------------------------------------------------
 Dataset: T2I-1M (/home/cc/ElasticIVF/hpdic/data/t2i/t2i_base_1M.fbin)
----------------------------------------------------------
  -> Loaded: N=1000000, D=200
  [1] GPU Flat (GpuIndexFlatL2)
      Add: 6033681 vec/s
      Del: 1181.51 ms (Roundtrip)
  [2] CPU HNSW (IndexHNSWFlat, M=32)
      Add: 25431 vec/s
      Del: N/A (Not Supported by HNSW)
  [3] SIVF (GpuIndexSIVF)

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   200 -> 238471
  > Data Buffer: 1500000 -> 7631072 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
Clustering 50000 points in 200D to 1024 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.03 s
  Iteration 19 (0.32 s, search 0.24 s): objective=22642.1 imbalance=1.206 nsplit=0       
[SIVF::train] GPU K-Means complete. Quantizer populated with 1024 centroids.
      Add: 3843250 vec/s
      Del: 0.675055 ms (Native)

----------------------------------------------------------
 Dataset: GIST1M (/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs)
----------------------------------------------------------
  -> Loaded: N=1000000, D=960
  [1] GPU Flat (GpuIndexFlatL2)
      Add: 1266176 vec/s
      Del: 5940.69 ms (Roundtrip)
  [2] CPU HNSW (IndexHNSWFlat, M=32)

 */