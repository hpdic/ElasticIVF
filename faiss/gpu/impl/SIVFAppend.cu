#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

__device__ void write_to_slab(
        SlabManagerDevice& manager,
        idx_t* slab_id_buffer,
        int slab_idx,
        int slot_idx,
        int dim,
        const float* src_vec,
        idx_t user_id) {
    // Write Vector
    float* dst_vec =
            manager.slab_data + (size_t)slab_idx * 32 * dim + slot_idx * dim;
    for (int d = 0; d < dim; ++d)
        dst_vec[d] = src_vec[d];

    // Write ID
    size_t physical_id_idx = (size_t)slab_idx * 32 + slot_idx;
    slab_id_buffer[physical_id_idx] = user_id;

    // Write ATT
    uint64_t coord = ((uint64_t)slab_idx << 32) | (uint64_t)slot_idx;
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    att_ptr[user_id] = coord;

    __threadfence();
    // Enable bit in bitmap
    atomicOr(
            &(manager.slab_metadata[slab_idx].validity_bitmap),
            (1u << slot_idx));
}

__global__ void sivf_append_kernel(
        SlabManagerDevice manager,
        int* list_heads,
        idx_t* slab_ids,
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
    const float* src_vec = vecs + (size_t)i * dim;

    int attempts = 0;
    while (attempts < 1000) {
        attempts++;
        volatile int* heads_ptr = list_heads;
        int curr_head = heads_ptr[cluster_id];

        // 1. Try to append to existing head
        if (curr_head != -1) {
            SlabMetadata* md = &manager.slab_metadata[curr_head];
            int old_count = md->valid_count;
            if (old_count < 32) {
                int assumed = old_count;
                if (atomicCAS(&(md->valid_count), assumed, assumed + 1) ==
                    assumed) {
                    write_to_slab(
                            manager,
                            slab_ids,
                            curr_head,
                            assumed,
                            dim,
                            src_vec,
                            user_id);
                    return;
                }
                continue; // Retry
            }
        }

        // 2. Allocate new slab
        int free_idx = atomicSub(manager.free_list_top, 1);
        if (free_idx <= 0) {
            atomicAdd(manager.free_list_top, 1);
            return; // OOM (should effectively never happen with our resize)
        }
        int new_slab = manager.free_list[free_idx - 1];

        // Initialize metadata safely
        SlabMetadata* new_md = &manager.slab_metadata[new_slab];
        new_md->valid_count = 1;
        new_md->validity_bitmap = 0;

        // [CRITICAL] Use volatile cast on the STRUCT, not int*, to ensure
        // correct offset/alignment
        ((volatile SlabMetadata*)new_md)->next_slab_idx = curr_head;

        __threadfence();

        // 3. Publish Head
        if (atomicCAS(&list_heads[cluster_id], curr_head, new_slab) ==
            curr_head) {
            write_to_slab(
                    manager, slab_ids, new_slab, 0, dim, src_vec, user_id);
            return;
        }

        // Else: CAS failed. DO NOT return 'new_slab' to free list.
        // Leak it to prevent cycles. Loop again to try with new head.
    }
}

void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        idx_t* slab_ids,
        int num_vecs,
        int dim,
        const idx_t* assignments,
        const float* vecs,
        const idx_t* ids,
        cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_vecs + threads - 1) / threads;
    sivf_append_kernel<<<blocks, threads, 0, stream>>>(
            manager,
            list_heads,
            slab_ids,
            num_vecs,
            dim,
            assignments,
            vecs,
            ids);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss