/**
 * @file faiss/gpu/GpuIndexSIVF.h
 * @brief Header definition for GpuIndexSIVF, a GPU-resident inverted file index
 * supporting dynamic updates (insertion/deletion) via Slab memory management.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 *
 * @details This class extends GpuIndexIVF to support dynamic updates using a
 * Slab-based memory management system. It provides efficient insertion and
 * deletion of vectors while maintaining high search performance on the GPU. The
 * implementation includes overrides for training, adding vectors, searching,
 * and removing vectors by ID. It also manages the state of the SlabManager and
 * maintains the head slab for each inverted list to facilitate dynamic updates.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/utils/DeviceVector.cuh>
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
        int nlist, // Number of clusters (inverted lists)
        faiss::MetricType metric = faiss::METRIC_L2,
        GpuIndexIVFConfig config = GpuIndexIVFConfig());

    ~GpuIndexSIVF() override;

    void initSlabManager(size_t max_vectors, size_t slab_pool_size);

    // =======================================================
    // Public Overrides
    // =======================================================

    void train(idx_t n, const float* x) override;

    // From IndexIVF.h
    size_t remove_ids(const faiss::IDSelector& sel) override;

    // Mandatory virtual implementations for state management
    void reset() override;
    void updateQuantizer() override;

   protected:
   
    // =======================================================
    // Protected Overrides
    // =======================================================

    // Matching base class signature strictly for addImpl_ and searchImpl_
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;
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

    DeviceVector<idx_t>*
        slab_id_buffer_; // Dedicated buffer for storing IDs within each Slab
};

} // namespace gpu
} // namespace faiss
