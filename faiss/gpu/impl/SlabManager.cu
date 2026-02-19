/**
 * @file faiss/gpu/impl/SlabManager.cu
 * @brief Implementation of the SlabManager handling GPU memory allocation.
 * @author Dongfang Zhao <dzhao@uw.edu>
 * @date 2026-02-16
 *
 * @details This file handles the allocation, initialization, and lifecycle
 * management of GPU memory resources required for the SIVF index structure,
 * including slab metadata, vector data storage, and the address translation
 * table.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

// =========================================================
// Kernel Definitions
// =========================================================

/**
 * @brief Initializes the memory pool's free list and stack pointer.
 *
 * @details
 * This kernel performs a parallel initialization of the slab allocator state:
 * 1. Identity Mapping: Fills `free_list` such that `free_list[i] = i`.
 * This means initially, every physical slab (from 0 to pool_size-1) is
 * marked as "available".
 * 2. Stack Initialization: Sets the `free_list_top` to `pool_size`.
 * The allocator operates as a customized stack:
 * - Allocation: Decrements top (atomicSub) -> pops from the end.
 * - Deallocation: Increments top (atomicAdd) -> pushes back to the end.
 *
 * @param[out] free_list      Device array to store available slab IDs.
 * @param[out] free_list_top  Device pointer to the stack counter (scalar).
 * @param[in]  pool_size      Total number of slabs in the pool.
 *
 * @note This is a global kernel to allow direct launch configurations
 * (Grid/Block) independent of the host class structure.
 */
__global__ void init_free_list_kernel(
    int* free_list,
    int* free_list_top,
    int pool_size) {

    // Calculate global thread index (Standard CUDA pattern)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: Ensure we don't write beyond the allocated pool size
    if (idx < pool_size) {
        free_list[idx] = idx;
    }

    // Single-thread initialization for the global stack pointer
    // This is very much like MPI's "rank 0" doing global initialization. We can
    // afford this slight serialization here since it's a one-time setup cost
    // and avoids the complexity of atomic operations for this step. We set it
    // to 'pool_size' because the allocator consumes from the top down.
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

    // Retrieve the CUDA stream from the resource manager.
    // This stream acts as the execution queue, ensuring that memory allocation,
    // zero-initialization (memset), and the kernel launch occur in strict
    // sequence on the GPU without blocking the host CPU.
    auto stream = res->getDefaultStream(device);

    // =========================================================
    // GPU Memory Allocation (Async via Stream)
    // =========================================================
    // The resize() calls below trigger underlying allocations (e.g.,
    // cudaMalloc) and are ordered on the provided stream to ensure
    // serialization.

    // 1. Slab Metadata Pool (Headers)
    // Stores control information for each slab (e.g., num_vectors,
    // next_slab_idx). Mapping: 1-to-1 correspondence with physical slabs.
    slab_metadata_.resize(slab_pool_size, stream);

    // 2. Vector Data Pool (Payload)
    // A monolithic contiguous buffer storing the actual raw vector data.
    // Layout: Flattened array [Total Slabs * Capacity per Slab * Vector
    // Dimension].
    // @note: Cast to (size_t) prevents integer overflow during the large size
    // calculation.
    slab_data_.resize(
        (size_t)slab_pool_size * SIVF_SLAB_CAPACITY * dim, stream);

    // 3. Address Translation Table (Indirection Layer)
    // Maps a logical user vector ID [0, max_vectors) to its physical location
    // (Slab ID + Offset). This indirection enables efficient updates and
    // deletes.
    address_table_.resize(max_vectors, stream);

    // 4. Free List (Allocator Stack)
    // A pre-allocated stack containing the indices of all currently available
    // slabs. Initially populates with [0, 1, ..., slab_pool_size-1].
    free_list_.resize(slab_pool_size, stream);

    // 5. Stack Pointer (Atomic Counter)
    // A single integer acting as the top-of-stack pointer for the free list.
    // Managed via atomicAdd/atomicSub in kernels.
    free_list_top_.resize(1, stream);

    int block_size = 256;
    int grid_size = ((int)slab_pool_size + block_size - 1) / block_size;

    // Initialization: Set Address Table to INVALID state (0xFF).
    // This ensures that lookups for uninserted IDs return INVALID_COORD
    // instead of a false positive index (e.g., 0).
    CUDA_VERIFY(cudaMemsetAsync(
        address_table_.data(), 0xFF, max_vectors * sizeof(uint64_t), stream));

    // Initialization: Populate the free list with available slab indices.
    init_free_list_kernel<<<grid_size, block_size, 0, stream>>>(
        free_list_.data(), free_list_top_.data(), (int)slab_pool_size);

    CUDA_TEST_ERROR();
}

SlabManager::~SlabManager() {
    // Resources are automatically released by DeviceVector destructors
}

/// @brief Constructs a lightweight device view for GPU kernel execution.
///
/// @details This function acts as a bridge between Host (CPU) resource
/// management and
///          Device (GPU) execution context. It extracts raw pointers from the
///          host-side RAII containers (e.g., device_vectors) and packages them
///          into a POD (Plain Old Data) struct.
///
/// @note This operation is "Zero-Copy" regarding GPU memory allocation. It only
/// performs
///       a shallow copy of the pointer addresses (64-bit integers) and scalar
///       configurations. The returned object is designed to be passed by value
///       to CUDA __global__ kernels.
///
/// @return A fully initialized SlabManagerDevice object containing valid GPU
/// pointers.
SlabManagerDevice SlabManager::getDeviceView() {
  
    // 1. Initialize the lightweight POD struct
    SlabManagerDevice dev;

    // 2. Pointer Extraction (Raw GPU Addresses)
    // @note .data() returns the raw pointer to the underlying device memory.
    dev.slab_metadata =
        slab_metadata_.data();         ///< Ptr to Slab Metadata (Header info)
    dev.slab_data = slab_data_.data(); ///< Ptr to Vector Data (Payload)
    dev.address_table =
        address_table_.data(); ///< Ptr to Address Table (Indirection layer)
    dev.free_list = free_list_.data(); ///< Ptr to Memory Pool Stack (Free IDs)
    dev.free_list_top =
        free_list_top_.data(); ///< Ptr to Stack Pointer (Atomic counter)

    // 3. Context Configuration (Scalars)
    // @note These scalars define the physical boundaries for the kernel.
    dev.slab_pool_size =
        (int)slab_pool_size_; ///< Physical limit of the memory pool
    dev.dim = dim_;           ///< Dimension of the vectors

    return dev;
}

// Note: Device side member function implementations (e.g., allocSlab, freeSlab)
// have been moved to `SlabManager.cuh` as inline __device__ functions to
// ensure proper visibility and inlining during CUDA compilation.

} // namespace gpu
} // namespace faiss
