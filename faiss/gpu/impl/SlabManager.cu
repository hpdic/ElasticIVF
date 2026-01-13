/**
 * faiss/gpu/impl/SlabManager.cu
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

// Kernel: 初始化空闲链表 (这个还得留着，因为它是一个 global
// kernel，不是成员函数)
__global__ void init_free_list_kernel(
        int* free_list,
        int* free_list_top,
        int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pool_size) {
        free_list[idx] = idx;
    }
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

    slab_metadata_.resize(slab_pool_size, stream);
    slab_data_.resize(
            (size_t)slab_pool_size * SIVF_SLAB_CAPACITY * dim, stream);
    address_table_.resize(max_vectors, stream);
    free_list_.resize(slab_pool_size, stream);
    free_list_top_.resize(1, stream);

    int block_size = 256;
    int grid_size = ((int)slab_pool_size + block_size - 1) / block_size;

    // [关键修复] 必须将 Metadata 区域清零！
    // 否则 valid_count 是随机值，导致 Add Kernel 以为 slab 满了或者越界
    CUDA_VERIFY(cudaMemsetAsync(
            slab_metadata_.data(),
            0,
            slab_pool_size * sizeof(SlabMetadata),
            stream));

    // 初始化空闲链表
    init_free_list_kernel<<<grid_size, block_size, 0, stream>>>(
            free_list_.data(), free_list_top_.data(), (int)slab_pool_size);

    CUDA_TEST_ERROR();
}

SlabManager::~SlabManager() {}

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

// [注意] 删除了之前的 Device Implementation 部分
// 它们现在已经在 .cuh 里被定义为 inline __device__ 了

} // namespace gpu
} // namespace faiss