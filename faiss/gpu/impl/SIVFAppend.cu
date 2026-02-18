/**
 * @file faiss/gpu/impl/SIVFAppend.cu
 * @brief Implements the core logic for appending vectors to the SIVF index on
 * the GPU.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 * @details This file implements the core logic for appending vectors to the
 * SIVF index on the GPU. It includes the CUDA kernel for parallel appends and
 * the associated device functions.
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

/**
 * @brief Launches the CUDA kernel to append a batch of vectors to the SIVF
 * index.
 *
 * This host-side wrapper calculates the optimal grid and block dimensions and
 * dispatches the `sivf_append_kernel` asynchronously on the specified stream.
 * It manages the mapping of input vectors to their respective inverted lists
 * (slabs) based on pre-calculated cluster assignments.
 *
 * @param[in,out] manager      Reference to the device-side `SlabManagerDevice`.
 * Manages memory allocation for new slabs if a list is full.
 * @param[in,out] list_heads   Device pointer to an array of list head indices.
 * Updated if a new slab becomes the head of a list.
 * @param[in]     slab_ids     Device pointer to slab identifiers (used for
 * linking nodes).
 * @param[in]     num_vecs     The number of vectors in the current batch to
 * append.
 * @param[in]     dim          The dimensionality of each vector.
 * @param[in]     assignments  Device pointer to the cluster assignment indices
 * for each vector (size: `num_vecs`).
 * @param[in]     vecs         Device pointer to the flattened vector data
 * (size: `num_vecs * dim`).
 * @param[in]     ids          Device pointer to the unique global identifiers
 * (UIDs) corresponding to the vectors (size: `num_vecs`).
 * @param[in]     stream       The CUDA stream to use for asynchronous kernel
 * execution.
 *
 * @note This function assumes that `vecs`, `ids`, and `assignments` are already
 * resident in GPU memory.
 */
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

    // Define the thread block size.
    // 256 is a heuristic choice to maximize occupancy on most NVIDIA
    // architectures (balances register usage and shared memory per SM).
    int threads = 256;

    // Calculate the grid dimension (number of blocks).
    // Uses integer ceiling division `(N + T - 1) / T` to ensure there are
    // enough threads to cover all `num_vecs`, dealing with non-aligned batch
    // sizes.
    int blocks = (num_vecs + threads - 1) / threads;

    // Launch the element-wise kernel.
    // Each thread is responsible for appending exactly one vector to its
    // assigned list. The kernel is launched asynchronously on the provided
    // `stream` to overlap with other compute or memory operations.
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