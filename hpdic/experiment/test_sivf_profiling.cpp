/**
 * test_sivf_profiling.cpp
 * Profiling with NVIDIA Nsight Systems/Compute.
 * Adds NVTX markers to isolate Insert/Delete phases for hardware metric collection.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <cstring> // for memcpy

// --- NVTX for Profiling ---
#include <nvtx3/nvToolsExt.h>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include "sift/sift_loader.h"

using namespace faiss::gpu;

struct Config {
    std::string dataset_name;
    const char* file_path;
    size_t window_size = 200000;
    size_t batch_size = 10000;
    int steps = 5;              // Reduced steps for profiling (profilers generate huge logs)
    int nlist = 1024;
} cfg;

int main(int argc, char** argv) {
    // Default to SIFT1M
    cfg.dataset_name = "SIFT1M";
    cfg.file_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";

    if (argc > 1 && std::string(argv[1]) == "gist") {
        cfg.dataset_name = "GIST1M";
        cfg.file_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
        cfg.nlist = 1024; 
        cfg.window_size = 100000; 
        cfg.batch_size = 5000;
    }

    // 1. Data Loading
    size_t d, total_nb;
    std::cout << "[Loader] Reading " << cfg.dataset_name << "..." << std::endl;
    float* all_data = fvecs_read(cfg.file_path, &d, &total_nb);
    
    size_t required_data = cfg.window_size + cfg.steps * cfg.batch_size;
    std::vector<float> workspace_data;
    if (total_nb < required_data) {
        workspace_data.resize(required_data * d);
        for (size_t i = 0; i < required_data; ++i) {
            size_t src_idx = i % total_nb;
            std::memcpy(workspace_data.data() + i * d, all_data + src_idx * d, d * sizeof(float));
        }
    } else {
        workspace_data.resize(required_data * d);
        std::memcpy(workspace_data.data(), all_data, required_data * d * sizeof(float));
    }
    delete[] all_data;

    StandardGpuResources res;
    if (d > 500) res.setTempMemory(512 * 1024 * 1024);
    else res.setTempMemory(1024 * 1024 * 1024);

    faiss::IndexFlatL2 quantizer(d);
    std::vector<float> train_data(50000 * d);
    std::memcpy(train_data.data(), workspace_data.data(), 50000 * d * sizeof(float));

    std::cout << "\n=== HPDC Hardware Profiling Mode ===" << std::endl;
    std::cout << "Dataset: " << cfg.dataset_name << " | Steps: " << cfg.steps << std::endl;

    // =========================================================
    // Round 1: Faiss Baseline (Target for PCIe Bottleneck Profiling)
    // =========================================================
    {
        std::cout << ">>> Running Baseline..." << std::endl;
        nvtxRangePushA("Baseline_Total_Session"); // Mark the whole session

        faiss::gpu::GpuIndexIVFFlat* gpu_index = new faiss::gpu::GpuIndexIVFFlat(&res, &quantizer, d, cfg.nlist, faiss::METRIC_L2);
        gpu_index->train(50000, train_data.data());

        std::vector<faiss::idx_t> initial_ids(cfg.window_size);
        std::iota(initial_ids.begin(), initial_ids.end(), 0);
        gpu_index->add_with_ids(cfg.window_size, workspace_data.data(), initial_ids.data());

        faiss::idx_t current_max_id = cfg.window_size;
        faiss::idx_t current_min_id = 0;

        for (int s = 0; s < cfg.steps; ++s) {
            float* batch_ptr = workspace_data.data() + (current_max_id * d);
            std::vector<faiss::idx_t> add_ids(cfg.batch_size);
            std::iota(add_ids.begin(), add_ids.end(), current_max_id);
            std::vector<faiss::idx_t> del_ids(cfg.batch_size);
            std::iota(del_ids.begin(), del_ids.end(), current_min_id);
            faiss::IDSelectorBatch selector(cfg.batch_size, del_ids.data());

            // --- PROFILE: INSERTION ---
            nvtxRangePushA("Baseline_Insert");
            gpu_index->add_with_ids(cfg.batch_size, batch_ptr, add_ids.data());
            cudaDeviceSynchronize();
            nvtxRangePop(); // End Baseline_Insert

            // --- PROFILE: DELETION (The PCIe Killer) ---
            nvtxRangePushA("Baseline_Delete_Roundtrip");
            
            faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(gpu_index);
            cpu_index->remove_ids(selector);
            delete gpu_index; 
            gpu_index = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(
                faiss::gpu::index_cpu_to_gpu(&res, 0, cpu_index));
            delete cpu_index;

            cudaDeviceSynchronize();
            nvtxRangePop(); // End Baseline_Delete_Roundtrip

            current_max_id += cfg.batch_size;
            current_min_id += cfg.batch_size;
        }
        delete gpu_index;
        nvtxRangePop(); // End Baseline_Total_Session
    }

    // =========================================================
    // Round 2: SIVF (Target for Compute/Memory Utilization)
    // =========================================================
    {
        std::cout << "\n>>> Running SIVF..." << std::endl;
        nvtxRangePushA("SIVF_Total_Session");

        faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, cfg.nlist, faiss::METRIC_L2, config);

        size_t cap = cfg.window_size + cfg.batch_size * 2; 
        sivf_index.initSlabManager(cap, d);
        sivf_index.train(50000, train_data.data());

        std::vector<faiss::idx_t> initial_ids(cfg.window_size);
        std::iota(initial_ids.begin(), initial_ids.end(), 0);
        sivf_index.add_with_ids(cfg.window_size, workspace_data.data(), initial_ids.data());

        faiss::idx_t current_max_id = cfg.window_size;
        faiss::idx_t current_min_id = 0;

        for (int s = 0; s < cfg.steps; ++s) {
            float* batch_ptr = workspace_data.data() + (current_max_id * d);
            std::vector<faiss::idx_t> add_ids(cfg.batch_size);
            std::iota(add_ids.begin(), add_ids.end(), current_max_id);
            std::vector<faiss::idx_t> del_ids(cfg.batch_size);
            std::iota(del_ids.begin(), del_ids.end(), current_min_id);
            faiss::IDSelectorBatch selector(cfg.batch_size, del_ids.data());

            // --- PROFILE: INSERTION ---
            nvtxRangePushA("SIVF_Insert");
            sivf_index.add_with_ids(cfg.batch_size, batch_ptr, add_ids.data());
            cudaDeviceSynchronize();
            nvtxRangePop(); // End SIVF_Insert

            // --- PROFILE: DELETION (The Fast One) ---
            nvtxRangePushA("SIVF_Delete");
            sivf_index.remove_ids(selector);
            cudaDeviceSynchronize();
            nvtxRangePop(); // End SIVF_Delete

            current_max_id += cfg.batch_size;
            current_min_id += cfg.batch_size;
        }
        nvtxRangePop(); // End SIVF_Total_Session
    }

    return 0;
}