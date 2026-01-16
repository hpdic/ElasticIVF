/**
 * faiss/gpu/impl/SlabManager.cu
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Implementation of the Host side Slab Memory Manager.
 * This file handles the allocation, initialization, and lifecycle management
 * of GPU memory resources required for the SIVF index structure, including
 * slab metadata, vector data storage, and the address translation table.
 */

#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

// =========================================================
// Kernel Definitions
// =========================================================

/**
 * Kernel: Initialize the free list.
 *
 * Populates the free list with sequential indices [0, pool_size - 1].
 * Note: This remains a global kernel rather than a member function to
 * facilitate direct invocation via CUDA launch semantics.
 */
__global__ void init_free_list_kernel(
        int* free_list,
        int* free_list_top,
        int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool_size) {
        free_list[idx] = idx;
    }
    // Initialize the stack pointer to the top of the pool
    if (idx == 0) {
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
    auto stream = res->getDefaultStream(device);

    // Allocation of device memory resources
    slab_metadata_.resize(slab_pool_size, stream);
    slab_data_.resize(
            (size_t)slab_pool_size * SIVF_SLAB_CAPACITY * dim, stream);
    address_table_.resize(max_vectors, stream);
    free_list_.resize(slab_pool_size, stream);
    free_list_top_.resize(1, stream);

    int block_size = 256;
    int grid_size = ((int)slab_pool_size + block_size - 1) / block_size;

    // Initialization: Set Address Table to INVALID state (0xFF).
    // This ensures that lookups for uninserted IDs return INVALID_COORD
    // instead of a false positive index (e.g., 0).
    CUDA_VERIFY(cudaMemsetAsync(
            address_table_.data(),
            0xFF,
            max_vectors * sizeof(uint64_t),
            stream));

    // Initialization: Populate the free list with available slab indices.
    init_free_list_kernel<<<grid_size, block_size, 0, stream>>>(
            free_list_.data(), free_list_top_.data(), (int)slab_pool_size);

    CUDA_TEST_ERROR();
}

SlabManager::~SlabManager() {
    // Resources are automatically released by DeviceVector destructors
}

SlabManagerDevice SlabManager::getDeviceView() {
    SlabManagerDevice dev;
    dev.slab_metadata = slab_metadata_.data();
    dev.slab_data = slab_data_.data();
    dev.address_table = address_table_.data();
    dev.free_list = free_list_.data();
    dev.free_list_top = free_list_top_.data();
    dev.slab_pool_size = (int)slab_pool_size_;
    dev.dim = dim_;
    return dev;
}

// Note: Device side member function implementations (e.g., allocSlab, freeSlab)
// have been moved to `SlabManager.cuh` as inline __device__ functions to
// ensure proper visibility and inlining during CUDA compilation.

} // namespace gpu
} // namespace faiss