/**
 * faiss/gpu/GpuIndexSIVF.h
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/utils/DeviceVector.cuh> // [修复] 必须加这个，否则 DeviceVector<int> 报错
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
    
    // [新增] 必须实现的虚函数
    void reset() override;
    void updateQuantizer() override;

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

    // 存储每个倒排链表当前的 "Head" Slab ID
    // 大小 = nlist。如果 list_heads_[i] == -1，说明第 i 个簇是空的。
    // 我们总是向 Head 插入数据 (或者你可以维护 Tail，这里简化为 Head/Active
    // Slab)
    DeviceVector<int>* list_heads_;

    DeviceVector<idx_t>* slab_id_buffer_; // 专门用来存每个 Slab 里的 ID
};

} // namespace gpu
} // namespace faiss