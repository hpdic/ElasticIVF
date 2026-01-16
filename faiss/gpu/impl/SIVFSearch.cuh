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