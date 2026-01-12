/**
 * faiss/gpu/impl/SIVFStructs.cuh
 * * Definition of data structures for SIVF (Streaming IVF) architecture.
 * 
 * Author: Dongfang Zhao (dzhao@cs.washington.edu)
 * * Date: January 2026
 */

#pragma once

#include <cstdint>

namespace faiss {
namespace gpu {

// 设定 Slab 的容量。
// 32 是一个非常好的数字，因为它正好等于 NVIDIA GPU 的一个 Warp Size。
// 这意味着一个 Warp 的线程可以并行处理一个 Slab，无需复杂的同步。
constexpr int SIVF_SLAB_CAPACITY = 32;

// 使用 -1 代表链表的结尾 (NULL)
constexpr int SIVF_NULL_SLAB = -1;

/**
 * SlabMetadata
 * 这是一个轻量级的结构，用于管理链表关系和 Bitmap。
 * * 我们不会把 float* codes 放在这里，而是放在一个巨大的全局内存池中。
 * 这里只存管理信息。
 */
struct SlabMetadata {
    // 链表指针：指向同一个倒排链表中的下一个 Slab 的索引 (Index)
    // 使用 int 索引而不是原始指针，方便在 Host/Device 间拷贝调试，且只有 4 字节
    int next_slab_idx;

    // Bitmap-based Lazy Eviction 的核心
    // 因为 CAPACITY 是 32，我们正好用一个 uint32_t 来表示所有槽位的有效性。
    // 1 = 有效, 0 = 已删除 (Lazy Deleted)
    uint32_t validity_bitmap;

    // 记录这个 Slab 里实际存了多少个有效向量 (用于快速统计)
    int valid_count;

    __device__ __host__ SlabMetadata()
            : next_slab_idx(SIVF_NULL_SLAB),
              validity_bitmap(0),
              valid_count(0) {}
};

/**
 * AddressTableEntry
 * 对应 Challenge 3: Reverse Mapping Overhead
 * * 我们需要把 (BlockID, SlotOffset) 压缩到一个 64 位整数中，
 * 以便构建 O(1) 的 GPU-Resident Address Table。
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
