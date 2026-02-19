/**
 * @file faiss/gpu/impl/SIVFAppend.cu
 * @brief Implements the core logic for appending vectors to the SIVF index on
 * the GPU.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 * 
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
 * Persist a single vector data and its metadata to the assigned slab and slot.
 *
 * Context: This function is executed by a SINGLE GPU THREAD.
 * Each thread is responsible for exactly one vector. Before entering this
 * function, the thread has already successfully reserved its unique
 * 'slot_idx' within 'slab_idx' through atomic reservation.
 *
 * @param manager SlabManagerDevice instance containing raw memory pointers.
 * @param slab_id_buffer Global array storing the mapping from physical location
 * to logical user ID.
 * @param slab_idx The specific memory block index allocated for this vector.
 * @param slot_idx The exact position (0 to 31) within the chosen slab.
 * @param dim The dimension of the vector.
 * @param src_vec Pointer to the actual float data of the vector.
 * @param user_id The logical ID assigned to this vector.
 */
__device__ void write_to_slab(
        SlabManagerDevice& manager,
        idx_t* slab_id_buffer,
        int slab_idx,
        int slot_idx,
        int dim,
        const float* src_vec,
        idx_t user_id) {

    // 1. Calculate physical offsets and copy vector data
    // The entire slab_data pool is a single flat 1D float array.
    // Base address for this slab: slab_idx * 32 (vectors per slab) * dim
    // Specific offset for this slot: slot_idx * dim
    float* dst_vec =
            manager.slab_data + (size_t)slab_idx * 32 * dim + slot_idx * dim;
    for (int d = 0; d < dim; ++d)
        dst_vec[d] = src_vec[d];

    // 2. Store Reverse Mapping (Physical -> Logical)
    // Record which user_id occupies this specific physical slot.
    // This is required during search to return the correct labels to the user.
    size_t physical_id_idx = (size_t)slab_idx * 32 + slot_idx;
    slab_id_buffer[physical_id_idx] = user_id;

    // 3. Update Address Translation Table (Logical -> Physical)
    // Pack two 32-bit integers (slab_idx and slot_idx) into a single 64-bit
    // integer. High 32 bits: slab_idx Low 32 bits: slot_idx This provides O(1)
    // lookup time for future delete or update operations.
    uint64_t coord = ((uint64_t)slab_idx << 32) | (uint64_t)slot_idx;
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    att_ptr[user_id] = coord;

    // 4. Memory Barrier (Strict Ordering)
    // Ensures all writes to dst_vec, slab_id_buffer, and att_ptr are physically
    // committed to global memory before proceeding. This is critical because
    // the next step involves setting the validity bit, which signals to other
    // threads that the slot is now occupied and safe to read. Without this
    // barrier, there could be a race condition where another thread sees the
    // validity bit set but reads stale or incomplete data.
    __threadfence();

    // 5. Atomic Commit
    // Mark the validity_bitmap to indicate this slot is now readable.
    // 1u << slot_idx creates a bitmask for the specific slot.
    atomicOr(
            &(manager.slab_metadata[slab_idx].validity_bitmap),
            (1u << slot_idx));
}

/**
 * Kernel: SIVF Parallel Append
 *
 * Handles concurrent insertion of vectors into inverted lists.
 * Uses a CAS based optimistic locking strategy to append to the current
 * head slab or link a new slab if the current one is full.
 *
 * @param manager The memory manager struct containing slab_data, free_list, and
 * metadata.
 * @param list_heads Global array of size nlist. Stores the head slab index for
 * each cluster.
 * @param slab_ids Global array mapping physical memory slots back to logical
 * user IDs.
 * @param num_vecs Total number of incoming vectors in this batch to be
 * inserted.
 * @param dim The dimensionality of each vector.
 * @param assignments Array of size num_vecs. Contains the target cluster ID (0
 * to nlist-1) for each vector.
 * @param vecs Flattened array containing the float data of all incoming
 * vectors.
 * @param ids Array of size num_vecs. Contains the logical user IDs
 * corresponding to the incoming vectors.
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
    
    // Calculate global thread ID. One thread processes one vector.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vecs)
        return;

    // Retrieve routing info and payload for this thread's assigned vector.
    int cluster_id = (int)assignments[i];
    idx_t user_id = ids[i];
    const float* src_vec = vecs + (size_t)i * dim;

    // Optimistic locking loop. Cap at 1000 to prevent infinite hangs.
    int attempts = 0;
    while (attempts < 1000) {
        attempts++;

        // Read the current head of the target linked list.
        // Volatile prevents the compiler from caching this value in a register.
        volatile int* heads_ptr = list_heads;
        int curr_head = heads_ptr[cluster_id];

        // =======================================================
        // Path 1: Attempt to append to the existing active slab
        // =======================================================
        if (curr_head != -1) {
            SlabMetadata* md = &manager.slab_metadata[curr_head];
            int old_count = md->valid_count;

            // Check if there is room in the current slab (capacity is 32)
            if (old_count < 32) {
                int assumed = old_count;

                // Attempt to reserve the slot.
                // If the value in memory is still 'assumed', change it to
                // 'assumed + 1'. This is a standard CAS loop to ensure thread
                // safety without locks.
                if (atomicCAS(&(md->valid_count), assumed, assumed + 1) ==
                    assumed) {
                    write_to_slab(
                        manager,
                        slab_ids,
                        curr_head,
                        assumed, // The slot index we just reserved
                        dim,
                        src_vec,
                        user_id);
                    return; // Done
                }

                // CAS failed. Another thread stole the slot.
                // Restart the loop to read the new state.
                continue;
            }
        }

        // =======================================================
        // Path 2: Allocate and link a new slab (Head is full or empty)
        // =======================================================
        
        // Atomically pop a fresh slab from the global memory pool
        int free_idx = atomicSub(manager.free_list_top, 1);
        if (free_idx <= 0) {
            atomicAdd(manager.free_list_top, 1); // Revert counter
            return; // OOM: Should be prevented by host-side pre-sizing
        }

        // We have secured a new slab index from the free list. Now we need to
        // initialize it and link it to the current head.
        int new_slab = manager.free_list[free_idx - 1];

        // Initialize metadata for the new slab
        SlabMetadata* new_md = &manager.slab_metadata[new_slab];
        new_md->valid_count = 1;
        new_md->validity_bitmap = 0;

        // Point the new slab's next pointer to the old head.
        // The volatile cast ensures this write is not reordered.
        ((volatile SlabMetadata*)new_md)->next_slab_idx = curr_head;

        // Force memory visibility.
        // The new slab must be fully formed in global memory before we publish
        // it.
        __threadfence();

        // =======================================================
        // Path 3: Publish the new slab as the list head
        // =======================================================

        // Attempt to swap the list head pointer.
        // If the head is STILL 'curr_head', replace it with 'new_slab'.
        if (atomicCAS(&list_heads[cluster_id], curr_head, new_slab) ==
            curr_head) {

            // We successfully became the new head. 
            // Write our data into slot 0 of this new slab.
            write_to_slab(
                    manager, slab_ids, new_slab, 0, dim, src_vec, user_id);
            return; // Done
        }

        // TODO Failure Handling:
        // Another thread beat us to it and updated list_heads[cluster_id].
        // Our 'new_slab' is now disconnected and orphaned.
        // We do not attempt to free it here to avoid ABA problems.
        // We simply loop around and try to insert into the new head.
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
