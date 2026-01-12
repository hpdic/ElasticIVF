/**
 * faiss/gpu/GpuIndexSIVF.h
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <vector>

namespace faiss {
namespace gpu {

class GpuResources;
class SlabManager;

class GpuIndexSIVF : public GpuIndexIVF {
   public:
    // 构造函数：参数名和类型严格匹配基类
    GpuIndexSIVF(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexIVFConfig config = GpuIndexIVFConfig());

    ~GpuIndexSIVF() override;

    void initSlabManager(size_t max_vectors, size_t slab_pool_size);

    // =======================================================
    // Public Overrides
    // =======================================================

    // [修正] 不写 Index::idx_t，直接用 idx_t
    void train(idx_t n, const float* x) override;

    size_t remove_ids(const faiss::IDSelector& sel) override;

   protected:
    // =======================================================
    // Protected Overrides
    // =======================================================

    // [核心修正]
    // 1. 函数名是 addImpl_ (带下划线)
    // 2. 类型是 idx_t (千万别写 Index::idx_t)
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    // [核心修正]
    // searchImpl_ 的参数 k 是 int，不是 idx_t
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   protected:
    SlabManager* slab_manager_;
    bool is_slab_initialized_;
};

} // namespace gpu
} // namespace faiss