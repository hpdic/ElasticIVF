/**
 * faiss/gpu/impl/SIVFAppend.cuh
 */

#pragma once

#include <faiss/Index.h> // [关键] 必须包含这个来定义 idx_t
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        int n,
        int d,
        const idx_t* assignments, // 现在编译器认识 idx_t 了
        const float* x,
        const idx_t* ids,
        cudaStream_t stream);

}
} // namespace faiss