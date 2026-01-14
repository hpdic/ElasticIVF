#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Tensor.cuh> // [修复] 必须是 .cuh
#include "GpuIndexSIVF.h"

namespace faiss {
namespace gpu {

constexpr uint64_t INVALID_COORD = 0xFFFFFFFFFFFFFFFFULL;

__global__ void sivf_delete_kernel(
        SlabManagerDevice manager,
        const idx_t* ids_to_remove,
        int num_ids,
        int* deleted_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ids)
        return;

    idx_t target_id = ids_to_remove[idx];

    // [核心修复] 暴力强转：无视 AddressTableEntry 类型，直接按 uint64_t 读取
    uint64_t* att_ptr = (uint64_t*)manager.address_table;
    uint64_t coord = att_ptr[target_id];

    if (coord == INVALID_COORD)
        return;

    uint32_t slab_idx = (uint32_t)(coord >> 32);
    uint32_t slot_idx = (uint32_t)(coord & 0xFFFFFFFF);

    // [核心修复] SlabManagerDevice 是 Struct，没有 get_metadata() 方法
    // 直接通过数组下标访问
    SlabMetadata* md = &manager.slab_metadata[slab_idx];

    uint32_t mask = ~(1u << slot_idx);
    uint32_t old_bitmap = atomicAnd(&(md->validity_bitmap), mask);

    if ((old_bitmap >> slot_idx) & 1u) {
        atomicSub(&(md->valid_count), 1);
        atomicAdd(deleted_count, 1);

        // 同样强转指针来写回无效标记
        att_ptr[target_id] = INVALID_COORD;
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

    // [API 适配] 1. 获取当前设备 ID 用于构造 AllocInfo
    int device;
    cudaGetDevice(&device);

    // [API 适配] 2. 构造 AllocInfo (根据 DeviceVector 源码要求)
    AllocInfo info(AllocType::Other, device, MemorySpace::Device, stream);

    // [API 适配] 3. 正确构造 DeviceVector
    DeviceVector<idx_t> d_ids(res, info);

    // [API 适配] 4. 使用 append 替代 copyFrom (自动分配内存 + 拷贝)
    d_ids.append(ids.data(), ids.size(), stream);

    // [API 适配] 5. 计数器初始化
    DeviceVector<int> d_count(res, info);
    d_count.resize(1, stream); // 分配空间
    d_count.setAll(0, stream); // 初始化为 0

    int threads = 256;
    int blocks = (ids.size() + threads - 1) / threads;

    sivf_delete_kernel<<<blocks, threads, 0, stream>>>(
            slab_manager->getDeviceView(),
            d_ids.data(),
            ids.size(),
            d_count.data());

    CUDA_TEST_ERROR();

    // [API 适配] 6. 手动拷回结果 (DeviceVector 没有直接拷回指针的 copyTo)
    CUDA_VERIFY(cudaMemcpyAsync(
            h_count_out,
            d_count.data(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
}

} // namespace gpu
} // namespace faiss