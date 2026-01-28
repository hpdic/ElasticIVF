/**
 * File: test_sivf_pareto_v3.cpp
 * Date: 2026-01-28
 * Description: A/B Test for Pareto Frontier with SHARED QUANTIZER.
 * Trains on CPU to ensure both GPU indices use identical centroids,
 * isolating the architectural differences (Array vs Slab).
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <unordered_set>

// Faiss CPU Headers
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h> 

// Faiss GPU Headers
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include "sift/sift_loader.h"

using namespace faiss::gpu;

// --- Configuration ---
struct Config {
    std::string dataset_name;
    const char* base_path;
    const char* query_path;
    size_t window_size = 50000;  // 50k vectors
    int nlist = 1024;            // 1024 clusters
    int k = 10;                  // Recall@10
    int num_queries = 1000;      // Evaluate on 1000 queries
} cfg;

// --- Helper: Compute Recall ---
double compute_recall(int nq, int k, const faiss::idx_t* gt_labels, const faiss::idx_t* res_labels) {
    long long correct_matches = 0;
    for (int i = 0; i < nq; ++i) {
        std::unordered_set<faiss::idx_t> gt_set;
        for (int j = 0; j < k; ++j) {
            gt_set.insert(gt_labels[i * k + j]);
        }
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(res_labels[i * k + j])) correct_matches++;
        }
    }
    return (double)correct_matches / (nq * k);
}

int main(int argc, char** argv) {
    // 1. Config Setup
    cfg.dataset_name = "SIFT1M";
    cfg.base_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs";
    cfg.query_path = "/home/cc/ElasticIVF/hpdic/data/sift/sift_query.fvecs";

    if (argc > 1 && std::string(argv[1]) == "gist") {
        cfg.dataset_name = "GIST1M";
        cfg.base_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs";
        cfg.query_path = "/home/cc/ElasticIVF/hpdic/data/gist/gist_query.fvecs";
        cfg.num_queries = 500;
        cfg.nlist = 2048; // GIST often needs more clusters
    }

    // 2. Load Data
    size_t d, nb, nq;
    std::cout << "[Loader] Reading " << cfg.dataset_name << "..." << std::endl;
    float* all_data = fvecs_read(cfg.base_path, &d, &nb);
    float* query_data = fvecs_read(cfg.query_path, &d, &nq);
    int eval_nq = std::min((int)nq, cfg.num_queries);

    // Prepare active window data
    if (nb < cfg.window_size) {
        std::cerr << "Error: Dataset too small!" << std::endl;
        return 1;
    }
    std::vector<float> window_data(cfg.window_size * d);
    std::memcpy(window_data.data(), all_data, cfg.window_size * d * sizeof(float));
    delete[] all_data; // Free huge buffer

    // 3. Compute Ground Truth (CPU Brute Force)
    std::cout << "[GroundTruth] Computing exact CPU truth..." << std::endl;
    std::vector<faiss::idx_t> gt_labels(eval_nq * cfg.k);
    std::vector<float> gt_dists(eval_nq * cfg.k);
    {
        faiss::IndexFlatL2 cpu_gt_index(d);
        cpu_gt_index.add(cfg.window_size, window_data.data());
        cpu_gt_index.search(eval_nq, query_data, cfg.k, gt_dists.data(), gt_labels.data());
    }

    // 4. SHARED TRAINING (The Fix)
    // Train a CPU IVF index first. This guarantees a high-quality quantizer
    // that we can copy to both GPU indices.
    std::cout << "[Setup] Training shared Quantizer on CPU..." << std::endl;
    faiss::IndexFlatL2 cpu_quantizer(d);
    faiss::IndexIVFFlat cpu_ivf_index(&cpu_quantizer, d, cfg.nlist, faiss::METRIC_L2);
    
    // Train on the window data
    cpu_ivf_index.train(std::min((size_t)50000, cfg.window_size), window_data.data());
    std::cout << "[Setup] Shared Centroids Ready." << std::endl;

    StandardGpuResources res;
    res.setTempMemory(1024 * 1024 * 1024);

    // =========================================================
    // Benchmark 1: Baseline (Standard GPU IVF)
    // =========================================================
    {
        std::cout << "\n>>> [Baseline: GPU IVF] Initializing..." << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
        faiss::gpu::GpuIndexIVFFlat baseline_index(&res, d, cfg.nlist, faiss::METRIC_L2, config);
        
        // COPY TRAINED CENTROIDS (Crucial Step)
        baseline_index.copyFrom(&cpu_ivf_index);

        // Add Data
        std::cout << "    Adding " << cfg.window_size << " vectors..." << std::endl;
        std::vector<faiss::idx_t> ids(cfg.window_size);
        std::iota(ids.begin(), ids.end(), 0);
        baseline_index.add_with_ids(cfg.window_size, window_data.data(), ids.data());
        cudaDeviceSynchronize();

        // Search Loop
        std::vector<int> nprobes = {1, 5, 10, 20, 32, 40, 64, 80, 100, 128};
        std::vector<float> res_dists(eval_nq * cfg.k);
        std::vector<faiss::idx_t> res_labels(eval_nq * cfg.k);

        std::cout << "------------------------------------------" << std::endl;
        std::cout << "nprobe\tLatency(ms)\tRecall@" << cfg.k << std::endl;
        std::cout << "------------------------------------------" << std::endl;

        for (int nprobe : nprobes) {
            baseline_index.nprobe = nprobe; 
            
            // Warmup
            baseline_index.search(10, query_data, cfg.k, res_dists.data(), res_labels.data());
            cudaDeviceSynchronize();

            auto t0 = std::chrono::high_resolution_clock::now();
            baseline_index.search(eval_nq, query_data, cfg.k, res_dists.data(), res_labels.data());
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double avg_lat = std::chrono::duration<double, std::milli>(t1 - t0).count() / eval_nq;
            double recall = compute_recall(eval_nq, cfg.k, gt_labels.data(), res_labels.data());

            std::cout << nprobe << "\t" 
                      << std::fixed << std::setprecision(3) << avg_lat << "\t\t" 
                      << std::setprecision(4) << recall << std::endl;
        }
    }

    // =========================================================
    // Benchmark 2: SIVF (Ours)
    // =========================================================
    {
        std::cout << "\n>>> [SIVF (Ours)] Initializing..." << std::endl;
        faiss::gpu::GpuIndexIVFFlatConfig config; config.device = 0;
        faiss::gpu::GpuIndexSIVF sivf_index(&res, d, cfg.nlist, faiss::METRIC_L2, config);

        // COPY TRAINED CENTROIDS
        // This ensures SIVF uses the exact same clustering as Baseline.
        // It bypasses the SIVF internal fallback training logic.
        try {
            sivf_index.copyFrom(&cpu_ivf_index);
        } catch (...) {
            std::cerr << "[Warning] copyFrom failed, falling back to internal train..." << std::endl;
            sivf_index.train(std::min((size_t)50000, cfg.window_size), window_data.data());
        }

        // Initialize SIVF Memory Pool (Required for SIVF)
        size_t cap = cfg.window_size * 2; 
        sivf_index.initSlabManager(cap, d);

        // Add Data
        std::cout << "    Adding " << cfg.window_size << " vectors..." << std::endl;
        std::vector<faiss::idx_t> ids(cfg.window_size);
        std::iota(ids.begin(), ids.end(), 0);
        sivf_index.add_with_ids(cfg.window_size, window_data.data(), ids.data());
        cudaDeviceSynchronize();

        // Search Loop
        std::vector<int> nprobes = {1, 5, 10, 20, 32, 40, 64, 80, 100, 128};
        std::vector<float> res_dists(eval_nq * cfg.k);
        std::vector<faiss::idx_t> res_labels(eval_nq * cfg.k);

        std::cout << "------------------------------------------" << std::endl;
        std::cout << "nprobe\tLatency(ms)\tRecall@" << cfg.k << std::endl;
        std::cout << "------------------------------------------" << std::endl;

        for (int nprobe : nprobes) {
            sivf_index.nprobe = nprobe; 

            // Warmup
            sivf_index.search(10, query_data, cfg.k, res_dists.data(), res_labels.data());
            cudaDeviceSynchronize();

            auto t0 = std::chrono::high_resolution_clock::now();
            sivf_index.search(eval_nq, query_data, cfg.k, res_dists.data(), res_labels.data());
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double avg_lat = std::chrono::duration<double, std::milli>(t1 - t0).count() / eval_nq;
            double recall = compute_recall(eval_nq, cfg.k, gt_labels.data(), res_labels.data());

            std::cout << nprobe << "\t" 
                      << std::fixed << std::setprecision(3) << avg_lat << "\t\t" 
                      << std::setprecision(4) << recall << std::endl;
        }
    }

    delete[] query_data;
    return 0;
}