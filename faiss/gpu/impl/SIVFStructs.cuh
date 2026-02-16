/**
 * @file faiss/gpu/impl/SIVFStructs.cuh
 * @brief Definition of data structures for SIVF (Streaming IVF) architecture.
 * @author Dongfang Zhao <dzhao@uw.edu>
 * @date February 2026
 * @details This file defines the core data structures used in the SIVF index
 * architecture, including SlabMetadata for managing slab-level information and
 * AddressTableEntry for GPU-resident address translation. These structures are
 * designed to be compact and efficient for GPU memory usage, enabling
 * high-performance vector indexing and retrieval in the SIVF framework.
 */

#pragma once

#include <cstdint>

namespace faiss {
namespace gpu {

// Set the capacity of a Slab.
// 32 is chosen because it exactly matches the Warp Size of NVIDIA GPUs.
// This allows threads in a single Warp to process a Slab in parallel without complex synchronization.
constexpr int SIVF_SLAB_CAPACITY = 32;

// Use -1 to represent the end of the linked list (NULL)
constexpr int SIVF_NULL_SLAB = -1;

/**
 * SlabMetadata
 * This is a lightweight structure for managing linked list relationships and the Bitmap.
 * 
 * We do not store the float* codes here; instead, they are placed in a massive global memory pool.
 * This structure stores management metadata only.
 */
struct SlabMetadata {
    // Linked list pointer: Index of the next Slab in the same inverted list.
    // Using an int index instead of a raw pointer facilitates Host/Device copying/debugging and consumes only 4 bytes.
    int next_slab_idx;

    // Core of Bitmap-based Lazy Eviction.
    // Since CAPACITY is 32, we can perfectly use a uint32_t to represent the validity of all slots.
    // 1 = Valid, 0 = Deleted (Lazy Deleted)
    uint32_t validity_bitmap;

    // Tracks the count of valid vectors currently stored in this Slab (used for fast statistics).
    int valid_count;

    __device__ __host__ SlabMetadata()
            : next_slab_idx(SIVF_NULL_SLAB),
              validity_bitmap(0),
              valid_count(0) {}
};

/**
 * AddressTableEntry
 * This is called ATT in the paper: Locate the target slab.
 *
 * We need to compress (BlockID, SlotOffset) into a single 64-bit integer 
 * to construct an O(1) GPU-Resident Address Table.
 */
struct AddressTableEntry {
    uint64_t packed_address; // [ 32-bit SlabID | 32-bit SlotOffset ]

    __device__ __host__ void set(int slab_id, int slot_offset) {
        packed_address = ((uint64_t)slab_id << 32) | (uint32_t)slot_offset;
    }

    __device__ __host__ int get_slab_id() const {
        return (int)(packed_address >> 32);
    }

    __device__ __host__ int get_slot_offset() const {
        return (int)(packed_address & 0xFFFFFFFF);
    }
};

} // namespace gpu
} // namespace faiss
