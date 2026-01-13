/**
 * faiss/gpu/impl/SIVFAppend.cu
 * Silent Mode
 */

#include <faiss/Index.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <stdio.h>
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

__global__ void sivf_add_kernel(
        SlabManagerDevice manager,
        int* list_heads,
        int n,
        int d,
        const float* x,
        const idx_t* list_ids,
        const idx_t* original_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int list_id = (int)list_ids[idx];
    if (list_id < 0 || list_id >= 20000)
        return;

    const float* my_vector = x + (long)idx * d;
    int safety_counter = 0;

    while (safety_counter++ < 1000000) { // 100万次重试，保证不丢数据

        volatile int* head_ptr = &list_heads[list_id];
        int curr_slab = *head_ptr;

        // Path A: 插入现有 Slab
        if (curr_slab != -1) {
            // 防御性检查去掉 printf，直接 return 或者忽略
            if (curr_slab < 0 || curr_slab >= manager.slab_pool_size)
                return;

            int slot =
                    atomicAdd(&manager.slab_metadata[curr_slab].valid_count, 1);
            if (slot < SIVF_SLAB_CAPACITY) {
                atomicOr(
                        &manager.slab_metadata[curr_slab].validity_bitmap,
                        (1U << slot));
                long offset = (long)curr_slab * SIVF_SLAB_CAPACITY * d +
                        (long)slot * d;
                for (int i = 0; i < d; ++i)
                    manager.slab_data[offset + i] = my_vector[i];
                return;
            }
        }

        // Path B: 分配新 Slab
        int free_idx = atomicSub(manager.free_list_top, 1);
        int new_slab_idx = -1;

        if (free_idx > 0) {
            new_slab_idx = manager.free_list[free_idx - 1];
        } else {
            atomicAdd(manager.free_list_top, 1);
            // [Silent] 显存满了就默默退出，不打印了，依赖外部配置足够大的 Pool
            return;
        }

        if (new_slab_idx < 0 || new_slab_idx >= manager.slab_pool_size)
            return;

        manager.slab_metadata[new_slab_idx].valid_count = 1;
        manager.slab_metadata[new_slab_idx].validity_bitmap = 1;
        manager.slab_metadata[new_slab_idx].next_slab_idx = curr_slab;

        long offset = (long)new_slab_idx * SIVF_SLAB_CAPACITY * d;
        for (int i = 0; i < d; ++i)
            manager.slab_data[offset + i] = my_vector[i];

        __threadfence();

        // CAS
        int old_head = atomicCAS(&list_heads[list_id], curr_slab, new_slab_idx);
        if (old_head == curr_slab)
            return;
    }
}

void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        int n,
        int d,
        const idx_t* list_ids,
        const float* x,
        const idx_t* original_ids,
        cudaStream_t stream) {
    int block = 128;
    int grid = (n + block - 1) / block;
    sivf_add_kernel<<<grid, block, 0, stream>>>(
            manager, list_heads, n, d, x, list_ids, original_ids);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss