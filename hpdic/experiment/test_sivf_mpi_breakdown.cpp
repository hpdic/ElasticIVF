/**
 * test_sivf_mpi_breakdown.cpp
 * * 专门用于收集分布式搜索的延迟分解数据：
 * 1. Local Search (GPU Kernel)
 * 2. MPI Communication (Gather/Reduce)
 * 3. Total E2E Latency
 */

#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>

using namespace faiss;
using namespace faiss::gpu;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int d = 128;
    int nlist = 4096;
    int nprobe = 32;
    int k = 10;
    size_t nb_per_gpu = 300000;
    int nq = 10; // 这里的 nq 设置小一点，以观察单批次延迟

    // 绑定 GPU
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);

    StandardGpuResources res;
    GpuIndexIVFConfig config;
    config.device = rank % num_gpus;

    // 初始化索引
    GpuIndexSIVF index(&res, d, nlist, METRIC_L2, config);
    // index.initSlabManager(nb_per_gpu, nb_per_gpu/32 + nlist);
    size_t global_max_vectors = size * nb_per_gpu;
    index.initSlabManager(global_max_vectors, global_max_vectors/32 + nlist);    
    
    // 模拟训练和数据插入
    std::vector<float> train_data(nlist * 40 * d, 0.5f);
    index.train(nlist * 40, train_data.data());
    std::vector<float> xb(nb_per_gpu * d, 0.5f);
    std::vector<idx_t> ids(nb_per_gpu);
    for(size_t i=0; i<nb_per_gpu; ++i) ids[i] = rank * nb_per_gpu + i;
    index.add_with_ids(nb_per_gpu, xb.data(), ids.data());

    // 准备查询
    std::vector<float> xq(nq * d, 0.5f);
    index.nprobe = nprobe;

    // 热身
    std::vector<float> dist(nq * k);
    std::vector<idx_t> res_ids(nq * k);
    index.search(nq, xq.data(), k, dist.data(), res_ids.data());
    MPI_Barrier(MPI_COMM_WORLD);

    // ==========================================
    // 精细化计时开始
    // ==========================================
    double t_total_start = MPI_Wtime();

    // 1. 本地搜索计时 (GPU Kernel)
    cudaDeviceSynchronize();
    double t_comp_start = MPI_Wtime();
    index.search(nq, xq.data(), k, dist.data(), res_ids.data());
    cudaDeviceSynchronize();
    double t_comp_end = MPI_Wtime();

    // 2. 通信计时 (MPI Gather 模拟结果汇总)
    double t_comm_start = MPI_Wtime();
    std::vector<float> all_dist;
    if (rank == 0) all_dist.resize(nq * k * size);
    MPI_Gather(dist.data(), nq * k, MPI_FLOAT, 
               all_dist.data(), nq * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    double t_comm_end = MPI_Wtime();

    double t_total_end = MPI_Wtime();

    // 计算本地指标
    double local_comp = t_comp_end - t_comp_start;
    double local_comm = t_comm_end - t_comm_start;
    double local_total = t_total_end - t_total_start;

    // 汇总各节点数据到 Root
    double max_comp, max_comm, max_total;
    MPI_Reduce(&local_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n--- Latency Breakdown (NP=%d, NQ=%d, K=%d) ---\n", size, nq, k);
        printf("Local GPU Search: %10.6f s (%5.2f%%)\n", max_comp, (max_comp/max_total)*100);
        printf("MPI Communication: %10.6f s (%5.2f%%)\n", max_comm, (max_comm/max_total)*100);
        printf("Other (Sync/Merge):%10.6f s (%5.2f%%)\n", max_total - max_comp - max_comm, 
               ((max_total - max_comp - max_comm)/max_total)*100);
        printf("Total E2E Latency: %10.6f s\n", max_total);
    }

    MPI_Finalize();
    return 0;
}