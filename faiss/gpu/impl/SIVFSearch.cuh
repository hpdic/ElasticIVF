/**
 * @file faiss/gpu/impl/SIVFSearch.cuh
 * @brief Declaration of the SIVF Search Kernel for querying vectors.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 *
 * @details This file declares the interface for the SIVF search operation,
 * which performs approximate nearest neighbor search on the GPU using the
 * Slab-based IVF index structure. The search kernel is designed to efficiently
 * probe multiple slabs in parallel, leveraging the metadata and address mapping
 * provided by the SlabManagerDevice.
 */

#pragma once
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

void runSIVFSearch(
    SlabManagerDevice& manager,
    int* list_heads,
    idx_t* slab_ids,
    int num_queries,
    int dim,
    int k,
    int nprobe,
    const float* queries,
    const idx_t* coarse_ids,
    float* out_distances,
    idx_t* out_labels,
    cudaStream_t stream);

}
} // namespace faiss
