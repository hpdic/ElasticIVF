// faiss/gpu/impl/SIVFAppend.cu

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

// =========================================================
// Kernel: 插入向量并更新 ATT 表
// =========================================================
__global__ void sivf_append_kernel(
        SlabManagerDevice manager,
        int* list_heads,
        int num_vecs,
        int dim,
        const idx_t* assignments,
        const float* vecs,
        const idx_t* ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vecs)
        return;

    int cluster_id = (int)assignments[i];
    idx_t user_id = ids[i];

    // 1. 获取该簇的 Slab
    int current_slab_idx = list_heads[cluster_id];

    if (current_slab_idx == -1) {
        return;
    }

    // 2. 获取写入位置 (使用 valid_count 作为游标)
    SlabMetadata* md = &manager.slab_metadata[current_slab_idx];

    int slot_idx = atomicAdd(&(md->valid_count), 1);

    if (slot_idx >= 32) {
        atomicSub(&(md->valid_count), 1);
        return;
    }

    // 3. 写入向量数据
    float* dst_vec = manager.slab_data + (size_t)current_slab_idx * 32 * dim +
            slot_idx * dim;
    const float* src_vec = vecs + (size_t)i * dim;

    for (int d = 0; d < dim; ++d) {
        dst_vec[d] = src_vec[d];
    }

    // 4. 更新 ATT 表 (核心步骤)
    uint64_t coord = ((uint64_t)current_slab_idx << 32) | (uint64_t)slot_idx;
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    att_ptr[user_id] = coord;

    // 5. 更新位图
    atomicOr(&(md->validity_bitmap), (1u << slot_idx));
}

// =========================================================
// Kernel: 初始化 Cluster Heads
// =========================================================
__global__ void init_cluster_heads(
        SlabManagerDevice manager,
        int* list_heads,
        int nlist,
        int* free_list,
        int* free_list_top) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlist)
        return;

    if (list_heads[i] == -1) {
        int free_idx = atomicSub(free_list_top, 1);
        if (free_idx > 0) {
            int new_slab = free_list[free_idx - 1];
            list_heads[i] = new_slab;

            SlabMetadata* md = &manager.slab_metadata[new_slab];
            md->valid_count = 0;
            md->validity_bitmap = 0;
            md->next_slab_idx = -1;
        }
    }
}

// =========================================================
// Host Launcher
// =========================================================
// [修复] 参数1 改为引用传递 (SlabManagerDevice&)，匹配链接器签名
void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        int num_vecs,
        int dim,
        const idx_t* assignments,
        const float* vecs,
        const idx_t* ids,
        cudaStream_t stream) {
    int nlist = 4096;
    int threads_init = 256;
    int blocks_init = (nlist + threads_init - 1) / threads_init;

    init_cluster_heads<<<blocks_init, threads_init, 0, stream>>>(
            manager,
            list_heads,
            nlist,
            manager.free_list,
            manager.free_list_top);

    int threads = 256;
    int blocks = (num_vecs + threads - 1) / threads;

    // 注意：虽然宿主函数按引用接收，Kernel 依然按值传递 manager (这是正确的)
    sivf_append_kernel<<<blocks, threads, 0, stream>>>(
            manager, list_heads, num_vecs, dim, assignments, vecs, ids);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss