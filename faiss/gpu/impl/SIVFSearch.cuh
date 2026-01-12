/**
 * faiss/gpu/impl/SIVFSearch.cuh
 */
#pragma once

#include <faiss/Index.h> // for idx_t
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

void runSIVFSearch(
        SlabManagerDevice& manager,
        int* list_heads,
        int n,                // number of queries
        int d,                // dimension
        int k,                // top-k
        int nprobe,           // number of probes
        const float* queries, // [n, d]
        const idx_t* keys,    // [n, nprobe] (list IDs to search)
        float* out_distances, // [n, k]
        idx_t* out_indices,   // [n, k]
        cudaStream_t stream);

}
} // namespace faiss