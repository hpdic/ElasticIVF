/**
 * * File: faiss/gpu/impl/SIVFAppend.cu
 *
 * * Author: Dongfang Zhao
 * * Email:  dzhao@uw.edu
 *
 * * Description: Implementation of the parallel append kernel for SIVF.
 * This file handles the concurrent ingestion of vectors into the slab-based
 * linked list structure, utilizing atomic CAS operations for lock-free
 * synchronization and head pointer management.
 */

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/impl/SIVFAppend.cuh> 

namespace faiss {
namespace gpu {

/**
 * Helper: Persist vector data and metadata to the assigned slab slot.
 *
 * This function performs the actual data write after a slot has been
 * successfully reserved. It updates:
 * 1. The raw vector data buffer.
 * 2. The physical ID mapping.
 * 3. The Address Translation Table (ATT).
 * 4. The validity bitmap (atomically).
 */
__device__ void write_to_slab(
        SlabManagerDevice& manager,
        idx_t* slab_id_buffer,
        int slab_idx,
        int slot_idx,
        int dim,
        const float* src_vec,
        idx_t user_id) {
    // 1. Store Vector Data
    float* dst_vec =
            manager.slab_data + (size_t)slab_idx * 32 * dim + slot_idx * dim;
    for (int d = 0; d < dim; ++d)
        dst_vec[d] = src_vec[d];

    // 2. Store Physical ID
    size_t physical_id_idx = (size_t)slab_idx * 32 + slot_idx;
    slab_id_buffer[physical_id_idx] = user_id;

    // 3. Update Address Translation Table (ATT)
    // Encode slab index (high 32 bits) and slot index (low 32 bits)
    uint64_t coord = ((uint64_t)slab_idx << 32) | (uint64_t)slot_idx;
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    att_ptr[user_id] = coord;

    // Ensure global visibility of data writes before enabling the bitmap
    __threadfence();

    // 4. Atomically set the validity bit
    atomicOr(
            &(manager.slab_metadata[slab_idx].validity_bitmap),
            (1u << slot_idx)
    );
}

/**
 * Kernel: SIVF Parallel Append
 *
 * Handles concurrent insertion of vectors into inverted lists.
 * Uses a CAS-based optimistic locking strategy to append to the current
 * head slab or link a new slab if the current one is full.
 */
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

        // -------------------------------------------------------
        // Path 1: Attempt to append to the existing active slab
        // -------------------------------------------------------
        if (curr_head != -1) {
            SlabMetadata* md = &manager.slab_metadata[curr_head];
            int old_count = md->valid_count;

            if (old_count < 32) {
                int assumed = old_count;
                // Atomic reservation of the slot index
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
                    return; // Success
                }
                continue; // CAS failed (contention), retry immediately
            }
        }

        // -------------------------------------------------------
        // Path 2: Allocate and link a new slab
        // -------------------------------------------------------
        
        // Pop a fresh slab from the free list
        int free_idx = atomicSub(manager.free_list_top, 1);
        if (free_idx <= 0) {
            atomicAdd(manager.free_list_top, 1); // Revert counter
            return; // OOM: Should be prevented by host-side pre-sizing
        }
        int new_slab = manager.free_list[free_idx - 1];

        // Initialize metadata for the new slab
        SlabMetadata* new_md = &manager.slab_metadata[new_slab];
        new_md->valid_count = 1;
        new_md->validity_bitmap = 0;

        // Volatile cast ensures the write to next_slab_idx is 
        // strictly ordered before the publication CAS below.
        ((volatile SlabMetadata*)new_md)->next_slab_idx = curr_head;

        __threadfence();

        // -------------------------------------------------------
        // Path 3: Publish the new slab as the list head
        // -------------------------------------------------------
        if (atomicCAS(&list_heads[cluster_id], curr_head, new_slab) ==
            curr_head) {
            // Write to the first slot (index 0) of the new slab
            write_to_slab(
                    manager, slab_ids, new_slab, 0, dim, src_vec, user_id);
            return; // Success
        }

        // Failure Handling:
        // If CAS failed, another thread updated the head. 
        // The current `new_slab` is effectively orphaned (leaked) to avoid 
        // complex ABA issues or reclamation logic in this critical path.
        // We simply loop again and try to insert into the new head.
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