/**
 * @file faiss/gpu/GpuIndexSIVF.cu
 * @brief Implementation of the GpuIndexSIVF class.
 * @author Dongfang Zhao <(dzhao@uw.edu)>
 * @date February 2026
 * @details This file implements the GpuIndexSIVF class, a slab-based inverted
 * file index for GPU. It includes:
 * - Constructor and destructor for lifecycle management.
 * - Initialization of the SlabManager and associated buffers.
 * - Overrides for training, addition, search, and deletion workflows.
 * - A GPU-accelerated fallback for training using K-Means if the base class
 * training fails.
 * - A custom implementation of remove_ids that leverages GPU parallelism for
 * efficient deletion.
 */

#include <faiss/Clustering.h> // For manual K-Means fallback
#include <faiss/IndexFlat.h>  // For CPU temporary Index
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>         // HPDIC MOD
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h> 
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SIVFAppend.cuh>    // HPDIC MOD
#include <faiss/gpu/impl/SIVFSearch.cuh>    // HPDIC MOD
#include <faiss/gpu/impl/SlabManager.cuh>   // HPDIC MOD
#include <faiss/gpu/utils/DeviceTensor.cuh> 

namespace faiss {
namespace gpu {

// ===========================================================
// Construction & Destruction
// ===========================================================

GpuIndexSIVF::GpuIndexSIVF(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFConfig config)
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          slab_manager_(nullptr),
          is_slab_initialized_(false),
          list_heads_(nullptr),
          slab_id_buffer_(nullptr) {
    
    // Explicitly mark as untrained to ensure train() executes.
    // From Index.h
    this->is_trained = false;

    // Initialize the Quantizer (if not provided externally)
    // This quantizer will be trained in the train() method, and is used for
    // assigning vectors to inverted lists.
    if (!this->quantizer) {
        this->quantizer =
                new GpuIndexFlat(provider, dims, metric, config.flatConfig);
        this->own_fields = true;
    }
}

GpuIndexSIVF::~GpuIndexSIVF() {
    if (slab_manager_) {
        delete slab_manager_;
        slab_manager_ = nullptr;
    }
    if (list_heads_) {
        delete list_heads_;
        list_heads_ = nullptr;
    }
    if (slab_id_buffer_) {
        delete slab_id_buffer_;
        slab_id_buffer_ = nullptr;
    }
}

/**
 * @brief Initializes the Slab Memory Manager and associated GPU buffers.
 * This is the "bootstrapping" phase where we pre-allocate a large chunk of GPU memory
 * to serve as the pool for dynamic vector insertion.
 *
 * @param max_vectors The estimated initial number of vectors to support.
 * @param pool_size The estimated initial number of slabs (chunks).
 */
void GpuIndexSIVF::initSlabManager(size_t max_vectors, size_t pool_size) {

    // 0. Idempotency Check
    // Prevent double-initialization if called multiple times.
    if (is_slab_initialized_)
        return;

    int device = getCurrentDevice();
    auto stream = resources_->getDefaultStream(device);

    // =========================================================
    // 1. Capacity Planning (Heuristic Strategy)
    // =========================================================

    // Calculate how many slabs are needed strictly for the requested vectors.
    // Use ceiling division: (num + den - 1) / den
    // Each slab holds 32 vectors (defined by SIVF_SLAB_SIZE).
    size_t needed_slabs = (max_vectors + 31) / 32;

    // Aggressive Expansion Strategy:
    // We allocate 5x the needed amount plus a static buffer (4096).
    // 1. To accommodate dynamic growth without immediate reallocation.
    // 2. To mitigate fragmentation effects in the free list.
    // 3. GPU memory is high-bandwidth but allocation is high-latency, so we
    // prefer fewer large allocations.
    size_t safe_pool_size = std::max(pool_size, needed_slabs * 5 + 4096);

    // =========================================================
    // 2. Alignment & Safety Guard
    // =========================================================

    // Derive the total vector storage capacity directly from the slab count.
    size_t safe_max_vectors = safe_pool_size * 32;

    // printf("\n[HPDIC MEMORY FIX] Resizing:\n");
    // printf("  > Slab Pool:   %zu -> %zu\n", pool_size, safe_pool_size);
    // printf("  > Data Buffer: %zu -> %zu vectors (Avoids Overflow)\n\n",
    //        max_vectors,
    //        safe_max_vectors);

    // =========================================================
    // 3. Resource Allocation
    // =========================================================

    // A. Create the Manager
    // This constructor invokes cudaMalloc for the main 'slab_data' (float
    // buffer) and 'free_list' (int stack).
    slab_manager_ = new SlabManager(
            resources_.get(),
            device,
            safe_max_vectors,
            safe_pool_size,
            this->d);

    // B. Create the ID Buffer
    // Stores the global ID (idx_t) for every vector in the pool.
    // Size matches 'safe_max_vectors' to ensure 1:1 mapping with slab_data.
    slab_id_buffer_ = new DeviceVector<idx_t>(
            resources_.get(), makeDevAlloc(AllocType::Other, stream));
    slab_id_buffer_->resize(safe_max_vectors, stream);

    // Reset IDs to -1
    // This marks all slots as "empty" or "invalid" initially.
    CUDA_VERIFY(cudaMemsetAsync(
            slab_id_buffer_->data(),
            -1,
            safe_max_vectors * sizeof(idx_t),
            stream));

    // =========================================================
    // 4. Inverted List Initialization
    // =========================================================

    // C. Create List Heads
    // This is the "Directory" of the inverted index.
    // list_heads_[i] stores the Slab ID of the i-th cluster.
    if (list_heads_ == nullptr) {
        AllocInfo info(AllocType::Other, device, MemorySpace::Device, stream);
        list_heads_ = new DeviceVector<int>(resources_.get(), info);
    }

    FAISS_ASSERT(this->nlist > 0);
    list_heads_->resize(this->nlist, stream);

    if (list_heads_->data() == nullptr) {
        FAISS_THROW_MSG("DeviceVector allocation failed for list_heads");
    }

    // Reset Heads to -1
    // -1 indicates that the cluster is currently empty (no slabs assigned).
    CUDA_VERIFY(cudaMemsetAsync(
            list_heads_->data(), -1, this->nlist * sizeof(int), stream));

    // Mark initialization as complete
    is_slab_initialized_ = true;
}

// Declaration of external launcher function, from SIVFDeletion.cu
// TODO: This is so hacky. We should refactor the SIVF deletion logic into a
// proper class and expose a cleaner interface. This is a temporary workaround
// to get the functionality working without a major refactor.
void run_sivf_deletion(
        SlabManager* slab_manager,
        GpuResources* res,
        cudaStream_t stream,
        const std::vector<idx_t>& ids,
        int* h_count_out);

size_t GpuIndexSIVF::remove_ids(const faiss::IDSelector& sel) {
    
    FAISS_THROW_IF_NOT_MSG(is_slab_initialized_, "SIVF not initialized");

    const faiss::IDSelectorBatch* sel_batch =
            dynamic_cast<const faiss::IDSelectorBatch*>(&sel);

    std::vector<idx_t> ids_to_remove;

    if (sel_batch) {
        size_t n = sel_batch->set.size();
        ids_to_remove.reserve(n);
        for (auto id : sel_batch->set) {
            ids_to_remove.push_back(id);
        }
    } else {
        // Fallback or exception for non-batch selectors
        FAISS_THROW_MSG(
                "SIVF remove_ids currently ONLY supports IDSelectorBatch");
    }

    int num_removed = 0;
    auto stream = resources_->getDefaultStream(config_.device);

    // Invoke the logic defined in SIVFDeletion.cu
    run_sivf_deletion(
            slab_manager_,
            resources_.get(),
            stream,
            ids_to_remove,
            &num_removed);

    cudaStreamSynchronize(stream);

    return (size_t)num_removed;
}

// ===========================================================
// Protected Overrides (Implementation Details)
// ===========================================================

// addImpl_ (with underscore), parameter type idx_t
void GpuIndexSIVF::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(
            is_slab_initialized_,
            "SIVF not initialized. Call initSlabManager() first.");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "SIVF not trained");
    auto res = resources_.get();
    int device = getCurrentDevice();
    auto stream = res->getDefaultStream(device);

    // 1. Quantization
    DeviceVector<float> distances(
            res,
            AllocInfo(AllocType::Other, device, MemorySpace::Device, stream));
    distances.resize(n, stream);

    DeviceVector<idx_t> assignments(
            res,
            AllocInfo(AllocType::Other, device, MemorySpace::Device, stream));
    assignments.resize(n, stream);

    this->quantizer->search(n, x, 1, distances.data(), assignments.data());

    // 2. Parallel Append Kernel
    auto manager_view = slab_manager_->getDeviceView();

    runSIVFAppend(
            manager_view,
            list_heads_->data(), // [Modified] Use ->data()
            slab_id_buffer_->data(),
            (int)n,
            this->d,
            assignments.data(),
            x,
            ids,
            stream);

    // Increment total count manually as base class does not handle this.
    this->ntotal += n;
}

// addImpl_ (with underscore), parameter type idx_t
void GpuIndexSIVF::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_slab_initialized_, "SIVF not initialized");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "SIVF not trained");

    // 2. Handle nprobe (support dynamic override via params)
    int nprobe = this->nprobe;
    if (params) {
        const IVFSearchParameters* ivf_params =
                dynamic_cast<const IVFSearchParameters*>(params);
        if (ivf_params) {
            nprobe = ivf_params->nprobe;
        }
    }

    // 3. Coarse Quantization (First level search)
    // Use Faiss DeviceTensor for automatic memory management
    auto stream = resources_->getDefaultStream(getCurrentDevice());

    // Note: Faiss GPU internals often use int indexing. Casting n is safe
    // as batch size rarely exceeds 2 billion.
    DeviceTensor<float, 2, true> coarse_dis(
            resources_.get(),
            makeDevAlloc(AllocType::Other, stream),
            {(int)n, nprobe});
    DeviceTensor<idx_t, 2, true> coarse_ids(
            resources_.get(),
            makeDevAlloc(AllocType::Other, stream),
            {(int)n, nprobe});

    // Invoke Quantizer
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_ids.data());

    // ================== [DEBUG START] ==================
    // 1. Check if Quantizer is empty
    // if (quantizer->ntotal == 0) {
    //     printf("[ERROR] Quantizer is EMPTY! (ntotal=0). Did training fail?\n");
    // } else {
        // printf("[DEBUG] Quantizer ntotal = %ld\n", quantizer->ntotal);
    // }
    // ================== [DEBUG END] ====================

    // 4. Fine-grained Search (Invoke our Kernel)
    // Cast away constness to retrieve DeviceView
    SlabManager* mutable_mgr = const_cast<SlabManager*>(slab_manager_);

    SlabManagerDevice device_view = mutable_mgr->getDeviceView();
    runSIVFSearch(
            device_view,
            list_heads_->data(),
            slab_id_buffer_->data(),
            (int)n, // Cast idx_t -> int
            this->d,
            k,
            nprobe,
            x,
            coarse_ids.data(), // Only IDs needed, residual not computed yet
            distances,
            labels,
            stream
    );
}

void GpuIndexSIVF::reset() {
    // 1. Reset Quantizer
    if (quantizer) {
        quantizer->reset();
    }

    // 2. Reset List Heads (Set all to -1)
    if (is_slab_initialized_ && list_heads_) {
        int device = getCurrentDevice();
        auto stream = resources_->getDefaultStream(device);
        cudaMemsetAsync(
                list_heads_->data(), -1, this->nlist * sizeof(int), stream);
    }

    // TODO: SlabManager should also be reset (reset free_list_top).
    // Deferred for future optimization.
}

void GpuIndexSIVF::updateQuantizer() {
    // Callback invoked when the user replaces the CPU-side Quantizer.
    // No specific handling required for SIVF currently.
}

void GpuIndexSIVF::train(idx_t n, const float* x) {
    // 1. Attempt base class training
    GpuIndexIVF::train(n, x);

    // 2. Verify training success
    if (this->quantizer->ntotal == 0) {
        // printf("[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...\n");

        // === Optimized Fallback: GPU Accelerated Clustering ===

        // 1. Set clustering parameters
        faiss::Clustering clus(this->d, this->nlist);
        clus.verbose = true;
        clus.niter = 20; // High iteration count for quality, fast on GPU

        // Pass the GPU Quantizer directly!
        // Faiss will leverage the existing GPU Index to accelerate the assignment phase.
        // *this->quantizer is GpuIndexFlat, which natively supports GPU Search.
        this->quantizer->reset();
        clus.train(n, x, *this->quantizer);

        // 3. Populate the Quantizer with final centroids
        // Note: clus.train stores intermediate results in clus.centroids (CPU).
        // We must re-add them to the GPU quantizer.
        this->quantizer->reset();
        this->quantizer->add(this->nlist, clus.centroids.data());

        this->is_trained = true;

        // printf("[SIVF::train] GPU K-Means complete. Quantizer populated with %ld centroids.\n",
        //        this->quantizer->ntotal);
    }
}

} // namespace gpu
} // namespace faiss