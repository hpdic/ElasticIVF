/**
 * faiss/gpu/impl/SIVFSearch.cu
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SIVFSearch.cuh>
#include <faiss/gpu/utils/Limits.cuh>

namespace faiss {
namespace gpu {

// 计算 L2 距离平方
__device__ inline float l2_dist_sq(const float* a, const float* b, int d) {
    float dist = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// 维护 Top-K (简单的插入排序逻辑)
__device__ void update_topk(
        float* topk_dists,
        idx_t* topk_ids,
        int k,
        float dist,
        idx_t id) {
    // 1. 如果比堆顶（目前第k小的）还大，直接扔掉
    if (dist >= topk_dists[k - 1])
        return;

    // 2. 找到插入位置
    int i = k - 1;
    while (i > 0 && topk_dists[i - 1] > dist) {
        topk_dists[i] = topk_dists[i - 1];
        topk_ids[i] = topk_ids[i - 1];
        i--;
    }

    // 3. 插入
    topk_dists[i] = dist;
    topk_ids[i] = id;
}

__global__ void sivf_search_kernel(
        SlabManagerDevice manager,
        int* list_heads,
        int n,
        int d,
        int k,
        int nprobe,
        const float* queries,
        const idx_t* keys,
        float* out_distances,
        idx_t* out_indices) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= n)
        return;

    // 结果存放区 (Global Memory)
    float* my_dists = out_distances + (long)query_idx * k;
    idx_t* my_ids = out_indices + (long)query_idx * k;

    // 初始化 Top-K 为无穷大
    for (int i = 0; i < k; ++i) {
        my_dists[i] = Limits<float>::getMax();
        my_ids[i] = -1;
    }

    const float* my_query = queries + (long)query_idx * d;

    // 遍历 nprobe 个簇
    for (int p = 0; p < nprobe; ++p) {
        idx_t list_id = keys[(long)query_idx * nprobe + p];
        if (list_id < 0)
            continue;

        int curr_slab = list_heads[list_id];

        // 遍历链表 (Pointer Chasing)
        while (curr_slab != SIVF_NULL_SLAB) {
            // 读取元数据
            unsigned int bitmap =
                    manager.slab_metadata[curr_slab].validity_bitmap;

            // 遍历 Slab 内的 32 个槽位
            // TODO: 未来可以用 __ffs (find first set) 优化位图遍历
            for (int slot = 0; slot < SIVF_SLAB_CAPACITY; ++slot) {
                if (bitmap & (1U << slot)) {
                    long offset = (long)curr_slab * SIVF_SLAB_CAPACITY * d +
                            (long)slot * d;

                    // 算距离
                    float dist =
                            l2_dist_sq(my_query, manager.slab_data + offset, d);

                    // 这里的 ID 暂时用 "Slab Index | Slot" 的组合，或者直接
                    // offset 我们用 offset / d (即 vector index inside pool)
                    // 作为临时 ID
                    idx_t temp_id =
                            (idx_t)((long)curr_slab * SIVF_SLAB_CAPACITY +
                                    slot);

                    // 更新堆
                    update_topk(my_dists, my_ids, k, dist, temp_id);
                }
            }
            // 下一个 Slab
            curr_slab = manager.slab_metadata[curr_slab].next_slab_idx;
        }
    }
}

void runSIVFSearch(
        SlabManagerDevice& manager,
        int* list_heads,
        int n,
        int d,
        int k,
        int nprobe,
        const float* queries,
        const idx_t* keys,
        float* out_distances,
        idx_t* out_indices,
        cudaStream_t stream) {
    // 简单的配置：每个线程处理一个 Query
    int block = 128;
    int grid = (n + block - 1) / block;

    sivf_search_kernel<<<grid, block, 0, stream>>>(
            manager,
            list_heads,
            n,
            d,
            k,
            nprobe,
            queries,
            keys,
            out_distances,
            out_indices);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss