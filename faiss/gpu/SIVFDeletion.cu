/**
 * * File: faiss/gpu/impl/SIVFDeletion.cu
 *
 * * Author: Dongfang Zhao (dzhao@uw.edu)
 * * Date: February 2026
 *
 * * Description: Implementation of the GPU-resident deletion logic for SIVF.
 * This file contains the CUDA kernel for atomic bitmap invalidation and
 * the host-side wrapper to manage memory transfer and kernel execution.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Tensor.cuh> // [Fix] Must include .cuh for template definitions
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
 * Algorithm:
 * 1. Lookup the physical address (Slab ID + Slot ID) from the Address Table.
 * 2. If valid, access the corresponding SlabMetadata.
 * 3. Atomically clear the bit in `validity_bitmap`.
 * 4. If the bit was previously set, decrement `valid_count` and the global counter.
 * 5. Mark the Address Table entry as INVALID.
 */
__global__ void sivf_delete_kernel(
        SlabManagerDevice manager,
        const idx_t* ids_to_remove,
        int num_ids,
        int* deleted_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ids)
        return;

    idx_t target_id = ids_to_remove[idx];

    // Ignore AddressTableEntry struct type and read directly as uint64_t.
    // This avoids overhead and ensures atomic-compatible access patterns.
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    uint64_t coord = att_ptr[target_id];

    if (coord == INVALID_COORD)
        return;

    uint32_t slab_idx = (uint32_t)(coord >> 32);
    uint32_t slot_idx = (uint32_t)(coord & 0xFFFFFFFF);

    // SlabManagerDevice is a POD struct and lacks accessor methods like get_metadata().
    // We access the metadata array directly via the slab index.
    SlabMetadata* md = &manager.slab_metadata[slab_idx];

    uint32_t mask = ~(1u << slot_idx);
    
    // Atomic AND to clear the specific bit.
    // old_bitmap stores the state BEFORE the operation.
    uint32_t old_bitmap = atomicAnd(&(md->validity_bitmap), mask);

    // Check if the bit was previously 1 (i.e., we actually deleted something)
    if ((old_bitmap >> slot_idx) & 1u) {

        int old_count = atomicSub(&(md->valid_count), 1);
        atomicAdd(deleted_count, 1);

        // Invalidate the address table entry to prevent future access
        att_ptr[target_id] = INVALID_COORD;

        // Reclaim slab if empty
        // Note that this is different than the conventional tombstone approach:
        // We physically reclaim the slab back to the free pool when it becomes empty;
        // thus, future insertions can reuse it.
        // This avoids memory bloat in workloads with heavy deletions.
        // Tombstone, on the other hand, would keep the slab allocated.
        if (old_count == 1) {
            int old_top = atomicAdd(manager.free_list_top, 1);
            if (old_top < manager.slab_pool_size) {
                manager.free_list[old_top] = slab_idx;
            }
            md->validity_bitmap = 0;
        }

    }
}

void run_sivf_deletion(
        SlabManager* slab_manager,
        GpuResources* res,
        cudaStream_t stream,
        const std::vector<idx_t>& ids,
        int* h_count_out) {
    if (ids.empty()) {
        *h_count_out = 0;
        return;
    }

    // Retrieve current device ID for AllocInfo construction
    int device;
    cudaGetDevice(&device);

    // Construct AllocInfo
    // Required by DeviceVector constructors to specify memory location.
    AllocInfo info(AllocType::Other, device, MemorySpace::Device, stream);

    // Initialize DeviceVector for IDs
    DeviceVector<idx_t> d_ids(res, info);

    // Transfer Data
    // Use append() instead of copyFrom() as it handles allocation and copy.
    d_ids.append(ids.data(), ids.size(), stream);

    // Initialize Output Counter
    DeviceVector<int> d_count(res, info);
    d_count.resize(1, stream); // Allocate storage
    d_count.setAll(0, stream); // Initialize to 0

    int threads = 256;
    int blocks = (ids.size() + threads - 1) / threads;

    sivf_delete_kernel<<<blocks, threads, 0, stream>>>(
            slab_manager->getDeviceView(),
            d_ids.data(),
            ids.size(),
            d_count.data());

    CUDA_TEST_ERROR();

    // Retrieve Result
    // DeviceVector does not have a direct pointer copyTo method,
    // so we use raw cudaMemcpyAsync.
    CUDA_VERIFY(cudaMemcpyAsync(
            h_count_out,
            d_count.data(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
}

} // namespace gpu
} // namespace faiss