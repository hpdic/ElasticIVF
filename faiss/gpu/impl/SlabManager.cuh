/**
 * faiss/gpu/impl/SlabManager.cuh
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/SIVFStructs.cuh> // [修复] 引入定义，而不是重写定义
#include <faiss/gpu/utils/DeviceVector.cuh>

namespace faiss {
namespace gpu {

// [注意] 这里不再重复定义 SlabMetadata 和 AddressTableEntry
// 它们来自 SIVFStructs.cuh

// =========================================================
// Device View (包含 inline 实现以解决 Link Error)
// =========================================================
struct SlabManagerDevice {
    SlabMetadata* slab_metadata;
    float* slab_data;
    AddressTableEntry* address_table;

    int* free_list;
    int* free_list_top;

    int slab_pool_size;
    int dim;

    // Kernel 必须能看到的 inline 实现
    __device__ inline int allocate_slab() {
        int old_top = atomicSub(free_list_top, 1);
        if (old_top <= 0) {
            atomicAdd(free_list_top, 1);
            return SIVF_NULL_SLAB;
        }
        int slab_idx = free_list[old_top - 1];

        slab_metadata[slab_idx].next_slab_idx = SIVF_NULL_SLAB;
        slab_metadata[slab_idx].validity_bitmap = 0;
        slab_metadata[slab_idx].valid_count = 0;

        return slab_idx;
    }

    __device__ inline void free_slab(int slab_idx) {
        int old_top = atomicAdd(free_list_top, 1);
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