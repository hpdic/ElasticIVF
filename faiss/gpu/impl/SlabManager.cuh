/**
 * * File: faiss/gpu/impl/SlabManager.cuh
 *
 * * Author: Dongfang Zhao
 * * Email:  dzhao@uw.edu
 *
 * Header definition for the SlabManager, handling GPU memory allocation primitives.
 * Includes both the Host-side manager class and the Device-side view struct
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
    SlabMetadata* slab_metadata;
    float* slab_data;
    AddressTableEntry* address_table;

    int* free_list;
    int* free_list_top;

    int slab_pool_size;
    int dim;

    // Inline implementation required for visibility within Kernels
    __device__ inline int allocate_slab() {
        int old_top = atomicSub(free_list_top, 1);
        // Check for underflow (empty pool)
        if (old_top <= 0) {
            atomicAdd(free_list_top, 1); // Revert the counter
            return SIVF_NULL_SLAB;
        }
        int slab_idx = free_list[old_top - 1];

        // Reset metadata for the newly allocated slab
        slab_metadata[slab_idx].next_slab_idx = SIVF_NULL_SLAB;
        slab_metadata[slab_idx].validity_bitmap = 0;
        slab_metadata[slab_idx].valid_count = 0;

        return slab_idx;
    }

    __device__ inline void free_slab(int slab_idx) {
        int old_top = atomicAdd(free_list_top, 1);
        // Check for overflow (full pool)
        if (old_top < slab_pool_size) {
            free_list[old_top] = slab_idx;
        }
    }

    __device__ inline AddressTableEntry get_address(long vector_id) const {
        return address_table[vector_id];
    }

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
    int device_;
    size_t max_vectors_;
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