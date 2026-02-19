/**
 * @file faiss/gpu/SIVFDeletion.cu
 * @brief Implementation of the GPU-resident deletion logic for SIVF.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 *
 * @details This file implements the core deletion logic for the SIVF index on
 * the GPU. The deletion process is designed to be efficient and thread-safe,
 * leveraging atomic operations to manage concurrent modifications to the Slab
 * metadata and Address Table.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Tensor.cuh> // for template definitions
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/GpuIndexSIVF.h>

namespace faiss {
namespace gpu {

constexpr uint64_t INVALID_COORD = 0xFFFFFFFFFFFFFFFFULL;

/**
 * Kernel: SIVF Deletion
 *
 * Performs in-place logical deletion by atomically flipping the validity bit
 * in the Slab metadata.
 *
 * @param manager Device view of the slab memory manager.
 * @param ids_to_remove Array of logical vector IDs to delete.
 * @param num_ids Total number of IDs in the deletion batch.
 * @param deleted_count Pointer to a global counter tracking successful
 * deletions.
 */
__global__ void sivf_delete_kernel(
        SlabManagerDevice manager,
        const idx_t* ids_to_remove,
        int num_ids,
        int* deleted_count) {

    // 1. Thread Mapping
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ids)
        return;

    idx_t target_id = ids_to_remove[idx];

    // 2. Address Lookup
    // Read the 64 bit coordinate directly.
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    uint64_t coord = att_ptr[target_id];

    // Exit if the vector does not exist or was already deleted.
    if (coord == INVALID_COORD)
        return;

    // 3. Coordinate Unpacking
    // High 32 bits store the slab index. Low 32 bits store the slot index.
    uint32_t slab_idx = (uint32_t)(coord >> 32);
    uint32_t slot_idx = (uint32_t)(coord & 0xFFFFFFFF);

    SlabMetadata* md = &manager.slab_metadata[slab_idx];

    // 4. Atomic Invalidation
    // Create a mask where only the target slot bit is 0, all others are 1.
    uint32_t mask = ~(1u << slot_idx);

    // atomicAnd applies the mask and returns the bitmap state BEFORE the AND
    // operation. This allows us to check if the bit was previously set (valid)
    // or not.
    uint32_t old_bitmap = atomicAnd(&(md->validity_bitmap), mask);

    // 5. Verification and Cleanup
    // Check if the target bit was actually 1 before our operation.
    // This prevents double counting if multiple threads try to delete the same
    // ID.
    if ((old_bitmap >> slot_idx) & 1u) {

        // If our code runs in there, it implies that this thread is the winner
        // of the deletion for this slot. We can safely decrement the valid
        // count and increment the global deletion count.

        // Decrement the valid vector count for this slab.
        int old_count = atomicSub(&(md->valid_count), 1);

        // Increment the global successful deletion counter.
        atomicAdd(deleted_count, 1);

        // Overwrite the address table to prevent future lookups.
        att_ptr[target_id] = INVALID_COORD;

        // 6. Dynamic Slab Reclamation
        // If old_count was 1, our deletion just brought the count to 0.
        // The slab is now completely empty and can be returned to the memory
        // pool. This is different than a simple tombstone approach and allows
        // us to reuse slabs for future insertions, which is critical for
        // long-running applications with dynamic workloads.
        if (old_count == 1) {
            
            // A naive tombstone would NOT attempt to reclaim the slab, which
            // would lead to memory bloat over time. Instead, we push the slab
            // index back onto the free list stack for reuse.
            int old_top = atomicAdd(manager.free_list_top, 1);
            if (old_top < manager.slab_pool_size) {

                // Push the slab index back onto the free list stack (LIFO). 
                manager.free_list[old_top] = slab_idx;
            }

            // Reset the bitmap completely for future use.
            md->validity_bitmap = 0;
        }
    }
}

/**
 * Host wrapper function to launch the SIVF deletion kernel.
 *
 * This function manages the lifecycle of a deletion request from the CPU to the
 * GPU. It handles memory allocation for the target IDs, asynchronous data
 * transfer, kernel execution, and fetching the final deletion count back to the
 * host.
 *
 * @param slab_manager Pointer to the host side SlabManager instance.
 * @param res Pointer to the Faiss GpuResources object for memory management.
 * @param stream The CUDA stream to ensure asynchronous and ordered execution.
 * @param ids A standard C++ vector containing the logical IDs to be deleted.
 * @param h_count_out Pointer to host memory where the actual deleted count will
 * be stored.
 */
void run_sivf_deletion(
        SlabManager* slab_manager,
        GpuResources* res,
        cudaStream_t stream,
        const std::vector<idx_t>& ids,
        int* h_count_out) {

    // 1. Early Exit Check
    // If the input list is empty, return 0 immediately to save overhead.
    if (ids.empty()) {
        *h_count_out = 0;
        return;
    }

    // 2. Device Context Setup
    // Query the current active GPU device to configure memory allocation.
    int device;
    cudaGetDevice(&device);

    // Construct AllocInfo, which tells Faiss GpuResources where and how
    // to allocate the temporary memory (Device memory, specific stream).
    AllocInfo info(AllocType::Other, device, MemorySpace::Device, stream);

    // 3. Host to Device Data Transfer
    // Initialize a Faiss DeviceVector to hold the target IDs on the GPU.
    DeviceVector<idx_t> d_ids(res, info);

    // The append method automatically allocates the required capacity on the
    // device and asynchronously copies the data from the host pointer to the
    // GPU.
    d_ids.append(ids.data(), ids.size(), stream);

    // 4. Output Counter Initialization
    // We need a single integer on the GPU to accumulate the successful deletion
    // count.
    DeviceVector<int> d_count(res, info);
    d_count.resize(1, stream); // Allocate storage
    d_count.setAll(0, stream); // Initialize to 0

    // 5. Kernel Launch Configuration
    // Standard 1D grid configuration for a flat array mapping.
    // threads: 256 threads per block is a standard balance for occupancy.
    // blocks: Ceiling division ensures all items are covered even if ids.size()
    // is not a perfect multiple of 256.
    int threads = 256;
    int blocks = (ids.size() + threads - 1) / threads;

    // Dispatch the deletion kernel asynchronously on the specified stream.
    sivf_delete_kernel<<<blocks, threads, 0, stream>>>(
            slab_manager->getDeviceView(),
            d_ids.data(),
            ids.size(),
            d_count.data());

    // Catch any immediate kernel launch errors (e.g., invalid configuration).        
    CUDA_TEST_ERROR();

    // 6. Device to Host Data Transfer
    // Retrieve the final deleted count back to the CPU pointer.
    // We use raw cudaMemcpyAsync because DeviceVector lacks a specialized
    // method for copying a single element directly to a host pointer.
    CUDA_VERIFY(cudaMemcpyAsync(
            h_count_out,
            d_count.data(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
}

} // namespace gpu
} // namespace faiss
