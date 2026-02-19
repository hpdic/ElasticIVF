/**
 * @file faiss/gpu/impl/SIVFAppend.cuh
 * @brief GPU kernel for appending vectors to the SIVF index structure.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 *
 * @details This file contains the declaration of the SIVF Append Kernel, which
 * is responsible for inserting new vectors into the SIVF index structure on the
 * GPU. The kernel takes care of allocating slabs, updating metadata, and
 * ensuring thread-safe operations for concurrent insertions.
 */

#pragma once
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

void runSIVFAppend(
    SlabManagerDevice& manager,
    int* list_heads,
    idx_t* slab_ids,
    int num_vecs,
    int dim,
    const idx_t* assignments,
    const float* vecs,
    const idx_t* ids,
    cudaStream_t stream);

}
} // namespace faiss
