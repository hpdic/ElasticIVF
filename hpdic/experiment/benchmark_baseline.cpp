/**
 * benchmark_baseline.cpp
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Benchmark for Add and Remove operations on Faiss GPU IndexIVFFlat.
 * * This script simulates a sliding window workload:
 * 1. Adds a new batch of vectors.
 * 2. Removes the oldest batch of vectors.
 * * Since standard Faiss GPU indices do not support native removal, this benchmark
 * captures the cost of the "CPU Roundtrip" workaround (GPU -> CPU -> Remove -> GPU).
 */

#include <cstdio>
#include <vector>
#include <chrono>
#include <iostream>
#include <numeric>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>

using namespace std;
using namespace std::chrono;

/**
 * Helper: Read fvecs file format.
 * Format: [dim][v1...][dim][v2...]
 */
float* fvecs_read(const char* fname, size_t& d, size_t& n) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        perror("Failed to open file");
        return nullptr;
    }

    // 1. Read the dimension from the header of the first vector
    int d_int;
    if (fread(&d_int, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error reading dimension from %s\n", fname);
        fclose(f);
        return nullptr;
    }
    d = (size_t)d_int;

    // 2. Calculate the number of vectors (n) based on file size
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    rewind(f);

    // Each vector contains: 1 int (4 bytes) + d floats (4 * d bytes)
    size_t entry_size = sizeof(int) + d * sizeof(float);

    if (entry_size == 0 || file_size % entry_size != 0) {
        fprintf(stderr, "Error: File size is not a multiple of vector size.\n");
        fclose(f);
        return nullptr;
    }

    n = file_size / entry_size;

    // 3. Allocate memory and read data
    float* data = new float[n * d];

    for (size_t i = 0; i < n; i++) {
        // Skip the dimension header (int) for the current vector
        if (fseek(f, sizeof(int), SEEK_CUR) != 0) {
            perror("fseek failed");
            delete[] data;
            fclose(f);
            return nullptr;
        }

        // Read the float data
        if (fread(data + i * d, sizeof(float), d, f) != d) {
            fprintf(stderr, "Error reading vector %zu\n", i);
            delete[] data;
            fclose(f);
            return nullptr;
        }
    }

    fclose(f);
    return data;
}

int main() {
    // ==========================================
    // Configuration Parameters
    // ==========================================
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    const char* learn_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_learn.fvecs";
    int nlist = 1024;
    int window_size = 100000;
    int batch_size = 10000;
    int device_id = 0;

    // 1. Load Data
    size_t d, nb, nt;
    float* xb = fvecs_read(base_file, d, nb);
    float* xt = fvecs_read(learn_file, d, nt);
    printf("Data loaded. Dim: %zu, Base: %zu\n", d, nb);

    // 2. Initialize GPU Resources
    faiss::gpu::StandardGpuResources res;

    // 3. Build and Train Index (on CPU first)
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist, faiss::METRIC_L2);
    
    printf("Training index...\n");
    cpu_index.train(nt, xt);

    // 4. Move Index to GPU
    printf("Moving index to GPU...\n");
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = device_id;
    faiss::gpu::GpuIndexIVFFlat gpu_index(&res, &cpu_index, config);

    // 5. Pre-fill Window
    printf("Pre-filling window with %d vectors...\n", window_size);
    gpu_index.add(window_size, xb);

    long current_min_id = 0;

    printf("\n%-5s | %-10s | %-12s | %-15s | %-10s\n", "Step", "Add(ms)", "Remove(ms)", "Method", "Total(ms)");
    printf("-----------------------------------------------------------------\n");

    for (int step = 0; step < 10; step++) {
        size_t start_idx = window_size + step * batch_size;
        float* new_data = xb + start_idx * d;

        // Prepare IDs to remove (simulating FIFO queue)
        vector<faiss::idx_t> ids_to_remove(batch_size);
        iota(ids_to_remove.begin(), ids_to_remove.end(), current_min_id);

        // A. Benchmark Add
        auto t0 = high_resolution_clock::now();
        gpu_index.add(batch_size, new_data);
        auto t1 = high_resolution_clock::now();
        double add_time = duration<double, milli>(t1 - t0).count();

        // B. Benchmark Remove
        auto t2 = high_resolution_clock::now();
        string method_name = "Direct";
        
        try {
            // Attempt direct GPU removal (Expected to fail on Standard Faiss)
            faiss::IDSelectorBatch selector(batch_size, ids_to_remove.data());
            gpu_index.remove_ids(selector);
        } catch (const exception& e) {
            // Fallback: Simulate "CPU Roundtrip" overhead
            method_name = "CPU_Roundtrip";
            
            // 1. Download: GPU to CPU
            faiss::IndexIVFFlat* tmp_cpu_index = dynamic_cast<faiss::IndexIVFFlat*>(
                faiss::gpu::index_gpu_to_cpu(&gpu_index)
            );
            
            // 2. Modify: CPU Remove
            faiss::IDSelectorBatch selector(batch_size, ids_to_remove.data());
            tmp_cpu_index->remove_ids(selector);
            
            // 3. Upload: CPU to GPU (Reconstruct GPU index to simulate full overhead)
            gpu_index.~GpuIndexIVFFlat(); // Explicitly destroy old GPU index
            new (&gpu_index) faiss::gpu::GpuIndexIVFFlat(&res, tmp_cpu_index, config); // Placement new
            
            delete tmp_cpu_index;
        }
        
        auto t3 = high_resolution_clock::now();
        double remove_time = duration<double, milli>(t3 - t2).count();

        printf("%-5d | %-10.2f | %-12.2f | %-15s | %-10.2f\n", 
               step, add_time, remove_time, method_name.c_str(), add_time + remove_time);

        current_min_id += batch_size;
    }

    delete[] xb;
    delete[] xt;
    return 0;
}

/**
 * Compilation Command:
 * * g++ -O3 -std=c++17 -fopenmp benchmark_baseline.cpp -o benchmark_baseline.bin \
 * -I/home/cc/ElasticIVF \
 * -I/usr/local/cuda/include \
 * -L/home/cc/ElasticIVF/build/faiss \
 * -L/usr/local/cuda/lib64 \
 * -lfaiss \
 * -lopenblas \
 * -lcudart \
 * -lcublas
 */