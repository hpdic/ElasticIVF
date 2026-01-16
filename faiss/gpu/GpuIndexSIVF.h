/**
 * faiss/gpu/GpuIndexSIVF.h
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Header definition for GpuIndexSIVF, a GPU-resident inverted file index
 * supporting dynamic updates (insertion/deletion) via Slab memory management.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/utils/DeviceVector.cuh> // [Fix] Required for DeviceVector instantiation
#include <vector>

namespace faiss {
namespace gpu {

class GpuResources;
class SlabManager;

class GpuIndexSIVF : public GpuIndexIVF {
   public:
    // Constructor: Matches base class signature strictly
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

    // [Correction] Use idx_t directly (not Index::idx_t)
    void train(idx_t n, const float* x) override;

    size_t remove_ids(const faiss::IDSelector& sel) override;
    
    // [New] Mandatory virtual implementations for state management
    void reset() override;
    void updateQuantizer() override;

   protected:
    // =======================================================
    // Protected Overrides
    // =======================================================

    // [Core Fix]
    // 1. Function name is addImpl_ (with trailing underscore)
    // 2. Type must be idx_t
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    // [Core Fix]
    // Note: The parameter 'k' in searchImpl_ is int, not idx_t
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

    // Stores the current "Head" Slab ID for each inverted list.
    // Size = nlist. If list_heads_[i] == -1, the i-th cluster is empty.
    // We always insert new data into the Head (Active) Slab.
    DeviceVector<int>* list_heads_;

    DeviceVector<idx_t>* slab_id_buffer_; // Dedicated buffer for storing IDs within each Slab
};

} // namespace gpu
} // namespace faiss