/**
 * faiss/gpu/impl/SlabManager.cuh
 */
#pragma once

#include <faiss/gpu/impl/SIVFStructs.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>

namespace faiss {
namespace gpu {

struct SlabManagerDevice {
    SlabMetadata* slab_metadata;
    float* slab_data;
    AddressTableEntry* address_table;
    int* free_list;
    int* free_list_top;

    int slab_pool_size;
    int dim;

    __device__ int allocate_slab();
    __device__ void free_slab(int slab_idx);
    __device__ AddressTableEntry get_address(long vector_id) const;
    __device__ void update_address(
            long vector_id,
            int slab_idx,
            int slot_offset);
};

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

    // private: // 建议放在 private
    int device_;
    int dim_;
    size_t slab_pool_size_;
    size_t max_vectors_;

    // [关键修正] 必须在这里声明这些成员变量，否则 .cu 找不到
    DeviceVector<SlabMetadata> slab_metadata_;
    DeviceVector<float> slab_data_;
    DeviceVector<AddressTableEntry> address_table_;
    DeviceVector<int> free_list_;
    DeviceVector<int> free_list_top_;
};

} // namespace gpu
} // namespace faiss