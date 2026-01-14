#pragma once
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        idx_t* slab_ids, // <--- 必须加上这个！之前的版本这里没有，所以报错
        int num_vecs,
        int dim,
        const idx_t* assignments,
        const float* vecs,
        const idx_t* ids,
        cudaStream_t stream);

}
} // namespace faiss