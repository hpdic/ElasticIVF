/**
 * faiss/gpu/impl/SlabManager.cu
 *
 * Copyright (c) Dongfang Zhao (dzhao@uw.edu).
 * All rights reserved.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

// =========================================================
// Kernel: 初始化空闲链表
// =========================================================
__global__ void init_free_list_kernel(
        int* free_list,
        int* free_list_top,
        int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool_size) {
        // 初始状态：将所有 Slab ID (0 到 pool_size-1) 放入 Free List
        free_list[idx] = idx;
    }

    // 线程 0 负责初始化栈顶指针
    if (idx == 0) {
        // 栈顶指向数组末尾 (代表有 pool_size 个可用元素)
        *free_list_top = pool_size;
    }
}

// =========================================================
// Host Implementation (SlabManager)
// =========================================================

SlabManager::SlabManager(
        GpuResources* res,
        int device,
        size_t max_vectors,
        size_t slab_pool_size,
        int dim)
        : device_(device),
          max_vectors_(max_vectors),
          slab_pool_size_(slab_pool_size),
          dim_(dim),
          // [关键修正] 使用 AllocInfo (Type, Device, MemorySpace, Stream)
          // 初始化 DeviceVector
          slab_metadata_(
                  res,
                  AllocInfo(
                          AllocType::Other,
                          device,
                          MemorySpace::Device,
                          res->getDefaultStream(device))),
          slab_data_(
                  res,
                  AllocInfo(
                          AllocType::Other,
                          device,
                          MemorySpace::Device,
                          res->getDefaultStream(device))),
          address_table_(
                  res,
                  AllocInfo(
                          AllocType::Other,
                          device,
                          MemorySpace::Device,
                          res->getDefaultStream(device))),
          free_list_(
                  res,
                  AllocInfo(
                          AllocType::Other,
                          device,
                          MemorySpace::Device,
                          res->getDefaultStream(device))),
          free_list_top_(
                  res,
                  AllocInfo(
                          AllocType::Other,
                          device,
                          MemorySpace::Device,
                          res->getDefaultStream(device))) {
    // 获取当前流，用于后续的内存分配和内核启动
    auto stream = res->getDefaultStream(device);

    // [Step 1] 分配显存
    // DeviceVector 构造时大小为 0，必须调用 resize 分配实际物理显存
    slab_metadata_.resize(slab_pool_size, stream);

    // slab_data 存储实际向量。注意：Faiss 使用 int
    // 索引，如果显存非常大可能溢出，但 SIFT1M 场景足够 大小 = Slab数量 *
    // 每个Slab容量(32) * 向量维度
    slab_data_.resize(
            (size_t)slab_pool_size * SIVF_SLAB_CAPACITY * dim, stream);

    // 地址表：映射 VectorID -> 物理地址
    address_table_.resize(max_vectors, stream);

    // 空闲链表栈
    free_list_.resize(slab_pool_size, stream);
    free_list_top_.resize(1, stream); // 只需要 1 个 int 存储栈顶位置

    // [Step 2] 初始化 Free List
    int block_size = 256;
    int grid_size = ((int)slab_pool_size + block_size - 1) / block_size;

    init_free_list_kernel<<<grid_size, block_size, 0, stream>>>(
            free_list_.data(), free_list_top_.data(), (int)slab_pool_size);

    // 检查内核启动错误
    CUDA_TEST_ERROR();
}

SlabManager::~SlabManager() {
    // DeviceVector 会自动释放显存，无需手动操作
}

SlabManagerDevice SlabManager::getDeviceView() {
    SlabManagerDevice dev;
    // 获取原始指针传给 Kernel
    dev.slab_metadata = slab_metadata_.data();
    dev.slab_data = slab_data_.data();
    dev.address_table = address_table_.data();
    dev.free_list = free_list_.data();
    dev.free_list_top = free_list_top_.data();

    dev.slab_pool_size = (int)slab_pool_size_;
    dev.dim = dim_;
    return dev;
}

// =========================================================
// Device Implementation (SlabManagerDevice)
// =========================================================

__device__ int SlabManagerDevice::allocate_slab() {
    // 1. 原子操作：尝试从栈顶“弹出”一个 Slab
    // atomicSub 返回旧值。如果 old_top 是 5，减 1 后变成 4，返回 5。
    // 我们取 index = 4 (即 old_top - 1)
    int old_top = atomicSub(free_list_top, 1);

    // 2. 检查下溢 (Underflow)
    if (old_top <= 0) {
        // 栈空了，没有可用 Slab
        // 恢复栈顶指针 (Best-effort，并发下不一定准，但能防止一直负数)
        atomicAdd(free_list_top, 1);
        return SIVF_NULL_SLAB; // 返回错误码 -1
    }

    // 3. 获取 Slab ID
    int slab_idx = free_list[old_top - 1];

    // 4. 初始化元数据 (Reset)
    // 刚拿到的 Slab 可能包含旧数据，需要重置状态
    // 注意：这里没有加锁，假设调用者（通常是单个线程）负责后续写入
    slab_metadata[slab_idx].next_slab_idx = SIVF_NULL_SLAB;

    // 将 Bitmap 设为全 1 (0xFFFFFFFF)，表示所有 32 个槽位物理上可用
    // 实际有效性由 valid_count 控制，或者可以在分配时设为 0
    slab_metadata[slab_idx].validity_bitmap = 0xFFFFFFFF;
    slab_metadata[slab_idx].valid_count = 0;

    return slab_idx;
}

__device__ void SlabManagerDevice::free_slab(int slab_idx) {
    // 1. 原子操作：申请入栈空间
    int old_top = atomicAdd(free_list_top, 1);

    // 2. 将 ID 压入栈
    // 边界检查防止溢出
    if (old_top < slab_pool_size) {
        free_list[old_top] = slab_idx;
    }
}

__device__ AddressTableEntry
SlabManagerDevice::get_address(long vector_id) const {
    // O(1) 查表
    return address_table[vector_id];
}

__device__ void SlabManagerDevice::update_address(
        long vector_id,
        int slab_idx,
        int slot_offset) {
    // 构建压缩地址并写入
    AddressTableEntry entry;
    entry.set(slab_idx, slot_offset);
    address_table[vector_id] = entry;
}

} // namespace gpu
} // namespace faiss