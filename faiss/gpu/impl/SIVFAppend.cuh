#pragma once
#include <faiss/gpu/impl/SlabManager.cuh>
    
namespace faiss {
namespace gpu {

void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        int n,
        int d,
        const idx_t* assignments,
        const float* x,
        const idx_t* ids,
        cudaStream_t stream);

}
} // namespace faiss