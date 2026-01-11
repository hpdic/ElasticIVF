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

// 模拟 fvecs 文件读取（仅读取数据部分）
float* fvecs_read(const char* fname, size_t& d, size_t& n) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        perror("Failed to open file");
        return nullptr;
    }
    int d_int;
    fread(&d_int, sizeof(int), 1, f);
    d = d_int;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    n = size / (sizeof(float) * (d + 1));
    rewind(f);

    float* data = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        fseek(f, sizeof(int), SEEK_CUR); // 跳过维数标识
        fread(data + i * d, sizeof(float), d, f);
    }
    fclose(f);
    return data;
}

int main() {
    // 配置参数
    const char* base_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    const char* learn_file = "/home/cc/ElasticIVF/hpdic/data/sift/sift_learn.fvecs";
    int nlist = 1024;
    int window_size = 100000;
    int batch_size = 10000;
    int device_id = 0;

    // 1. 加载数据
    size_t d, nb, nt;
    float* xb = fvecs_read(base_file, d, nb);
    float* xt = fvecs_read(learn_file, d, nt);
    printf("Data loaded. Dim: %zu, Base: %zu\n", d, nb);

    // 2. 初始化 GPU 资源
    faiss::gpu::StandardGpuResources res;

    // 3. 构建并训练索引
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat cpu_index(&quantizer, d, nlist, faiss::METRIC_L2);
    
    printf("Training index...\n");
    cpu_index.train(nt, xt);

    // 4. 搬运到 GPU
    printf("Moving index to GPU...\n");
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = device_id;
    faiss::gpu::GpuIndexIVFFlat gpu_index(&res, &cpu_index, config);

    // 5. 预填充窗口
    printf("Pre-filling window with %d vectors...\n", window_size);
    gpu_index.add(window_size, xb);

    long current_min_id = 0;

    printf("\n%-5s | %-10s | %-12s | %-15s | %-10s\n", "Step", "Add(ms)", "Remove(ms)", "Method", "Total(ms)");
    printf("-----------------------------------------------------------------\n");

    for (int step = 0; step < 10; step++) {
        size_t start_idx = window_size + step * batch_size;
        float* new_data = xb + start_idx * d;

        // 准备待删除的 ID
        vector<faiss::idx_t> ids_to_remove(batch_size);
        iota(ids_to_remove.begin(), ids_to_remove.end(), current_min_id);

        // A. 测试 Add
        auto t0 = high_resolution_clock::now();
        gpu_index.add(batch_size, new_data);
        auto t1 = high_resolution_clock::now();
        double add_time = duration<double, milli>(t1 - t0).count();

        // B. 测试 Remove
        auto t2 = high_resolution_clock::now();
        string method_name = "Direct";
        
        try {
            faiss::IDSelectorBatch selector(batch_size, ids_to_remove.data());
            gpu_index.remove_ids(selector);
        } catch (const exception& e) {
            // 如果 GPU 直接删除失败（Faiss 默认不支持），进入 Roundtrip 模拟
            method_name = "CPU_Roundtrip";
            
            // 1. GPU to CPU
            faiss::IndexIVFFlat* tmp_cpu_index = dynamic_cast<faiss::IndexIVFFlat*>(
                faiss::gpu::index_gpu_to_cpu(&gpu_index)
            );
            
            // 2. CPU Remove
            faiss::IDSelectorBatch selector(batch_size, ids_to_remove.data());
            tmp_cpu_index->remove_ids(selector);
            
            // 3. CPU to GPU (这里我们重建 GPU 索引来模拟完整开销)
            gpu_index.~GpuIndexIVFFlat(); // 销毁旧索引
            new (&gpu_index) faiss::gpu::GpuIndexIVFFlat(&res, tmp_cpu_index, config);
            
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
g++ -O3 -std=c++11 benchmark_sivf.cpp -o benchmark_sivf \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lfaiss -lcudart
*/