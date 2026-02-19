/**
 * @file faiss/gpu/impl/SlabManager.cuh
 * @brief Header definition for the SlabManager handling GPU memory allocation.
 * @author Dongfang Zhao <dzhao@uw.edu>
 * @date   2026-02-16
 * 
 * @details Header definition for the SlabManager, handling GPU memory allocation 
 * primitives. Includes both the Host-side manager class and the Device-side view struct
 * with inline __device__ allocation logic.
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/SIVFStructs.cuh> // HPDIC MOD
#include <faiss/gpu/utils/DeviceVector.cuh>

namespace faiss {
namespace gpu {

// [Note] SlabMetadata and AddressTableEntry are not redefined here.
// They are imported from SIVFStructs.cuh.

// =========================================================
// Device View (Includes inline implementations to resolve Link Errors)
// =========================================================
struct SlabManagerDevice {

    /// @brief Pointer to the metadata array for each slab.
    /// Stores validity bitmaps, next pointers, and other slab-level info.
    /// Access: slab_metadata[slab_idx]
    SlabMetadata* slab_metadata;

    /// @brief Pointer to the main flat data buffer in GPU memory.
    /// Total size = slab_pool_size * dim.
    /// Layout: Flattened 1D array. Access via: slab_data[slab_idx * dim + offset]
    float* slab_data;

    /// @brief Global address mapping table (Page Table).
    /// Maps a logical vector_id to its physical location (slab_idx, slot_offset).
    /// Access: address_table[vector_id]
    AddressTableEntry* address_table;

    /// @brief Stack of available (free) slab indices.
    /// A pre-allocated array acting as a LIFO stack for memory allocation.
    int* free_list;

    /// @brief Pointer to the current top index of the free_list stack.
    /// WARNING: This points to a value in global memory.
    /// Must be accessed using atomicSub/atomicAdd for thread safety.
    int* free_list_top;

    /// @brief Total capacity of the memory pool (number of slabs).
    /// Used for boundary checks to prevent overflow/underflow.
    int slab_pool_size;

    /// @brief Dimensionality of the vectors (e.g., 128, 768).
    /// Used for stride calculation in the flattened data buffer.
    int dim;

    /**
     * @brief Allocates a new slab from the free list in a thread-safe manner.
     * * This function atomically decrements the free_list_top counter to reserve a slot.
     * If successful, it pops a slab index from the stack and initializes its metadata.
     * * @note Thread Safety: Uses atomicSub/atomicAdd to ensure safe concurrent access 
     * by thousands of GPU threads.
     * * @return int The index of the allocated slab, or SIVF_NULL_SLAB (-1) if the pool is empty (underflow).
     */    
    __device__ inline int allocate_slab() {

        // Atomic decrement to reserve a slot. 
        // Returns the value BEFORE decrementing (old_top).
        int old_top = atomicSub(free_list_top, 1);

        // Check for underflow (empty pool)
        // Critical Section: If multiple threads decrement below 0, we must revert.
        if (old_top <= 0) {
            atomicAdd(free_list_top, 1);    // Revert the counter
            return SIVF_NULL_SLAB;          // Allocation failed
        }

        // Retrieve the slab index from the stack (LIFO)
        // Access old_top - 1 because indices are 0-based
        int slab_idx = free_list[old_top - 1];

        // Reset metadata for the newly allocated slab
        // This ensures the slab is "clean" before use.
        slab_metadata[slab_idx].next_slab_idx = SIVF_NULL_SLAB;
        slab_metadata[slab_idx].validity_bitmap = 0;
        slab_metadata[slab_idx].valid_count = 0;

        return slab_idx;
    }

    /**
     * @brief Releases a slab index back to the free list (Stack Push).
     * @details This function atomically increments the free_list_top counter to reserve 
     * a slot in the free_list array, and then writes the slab_idx into that slot.
     * @param slab_idx The index of the slab to be freed. 
     * Must be a valid index [0, slab_pool_size-1].
     * @note Thread Safety: Uses atomicAdd to safely coordinate concurrent frees 
     * from multiple GPU threads.
     */
    __device__ inline void free_slab(int slab_idx) {

        // Atomically increment the stack pointer to reserve a spot.
        // Returns the value BEFORE incrementing (old_top), which is our write index.
        int old_top = atomicAdd(free_list_top, 1);

        // Check for overflow (Safety Guard).
        // Prevents writing out of bounds if the pool is somehow corrupted 
        // or if we try to free more slabs than the pool can hold.
        if (old_top < slab_pool_size) {
            free_list[old_top] = slab_idx;
        }
    }

    /**
     * @brief Retrieves the physical location (Slab ID + Slot Offset) for a logical vector ID.
     * @param vector_id The global index of the vector. 
     * @return AddressTableEntry The physical address. 
     * @warning No bounds checking. Caller must ensure vector_id < max_vectors.
     */
    __device__ inline AddressTableEntry get_address(long vector_id) const {
        return address_table[vector_id];
    }

    /**
     * @brief Updates the page table mapping for a specific vector.
     * @details Maps a logical `vector_id` to a physical `slab_idx` and `slot_offset`.
     * @param vector_id The global index to update.
     * @param slab_idx The physical slab index where data is stored.
     * @param slot_offset The specific slot within that slab.
     */
    __device__ inline void update_address(
            long vector_id,
            int slab_idx,
            int slot_offset) {
        AddressTableEntry entry;
        entry.set(slab_idx, slot_offset);
        address_table[vector_id] = entry;
    }
};

// =========================================================
// Host Manager
// =========================================================
class SlabManager {
public:
    SlabManager(
            GpuResources* res,
            int device,
            size_t max_vectors,
            size_t slab_pool_size,
            int dim);
    ~SlabManager();

    SlabManagerDevice getDeviceView();

public:
    // These two members are only used on the Host side
    int device_;
    size_t max_vectors_;

    // The following seven members are exactly mirrored in the 
    // SlabManagerDevice struct for direct GPU access.
    size_t slab_pool_size_;
    int dim_;

    DeviceVector<SlabMetadata> slab_metadata_;
    DeviceVector<float> slab_data_;
    DeviceVector<AddressTableEntry> address_table_;

    DeviceVector<int> free_list_;
    DeviceVector<int> free_list_top_;
};

} // namespace gpu
} // namespace faiss
