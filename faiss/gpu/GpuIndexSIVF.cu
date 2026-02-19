/**
 * @file faiss/gpu/GpuIndexSIVF.cu
 * @brief Implementation of the GpuIndexSIVF class.
 * @author Dongfang Zhao <(dzhao@uw.edu)>
 * @date February 2026
 * 
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
    // Use ceiling division: (num + len - 1) / len
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

/**
 * @brief The core implementation for adding vectors to the SIVF index.
 * * This overrides the standard GpuIndexIVF::addImpl_ to use our dynamic Slab
 * memory system.
 * * The process consists of two stages:
 * 1. Quantization: Find the nearest cluster (inverted list) for each input
 * vector.
 * 2. Parallel Append: concurrently write vectors into the appropriate Slabs.
 *
 * @param n Number of vectors to add.
 * @param x Pointer to the vector data (on Host or Device, Faiss handles this).
 * @param ids Pointer to the external IDs for these vectors.
 */
void GpuIndexSIVF::addImpl_(idx_t n, const float* x, const idx_t* ids) {

    // =========================================================
    // 0. Pre-flight Checks (Safety First)
    // =========================================================

    // Ensure the memory pool is allocated.
    FAISS_THROW_IF_NOT_MSG(
            is_slab_initialized_,
            "SIVF not initialized. Call initSlabManager() first.");

    // Ensure the quantizer (centroids) is ready.
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "SIVF not trained");

    // Get current GPU context and stream for subsequent operations.
    auto res = resources_.get();
    int device = getCurrentDevice();
    auto stream = res->getDefaultStream(device);

    // =========================================================
    // 1. Stage 1: Quantization
    // =========================================================
    // We need to know which inverted list (cluster) each vector belongs to.
    // This is done by searching for the nearest centroid in the quantizer.

    // A. Allocate temporary buffers for quantization results

    // 'distances' is needed by the API but not used for insertion logic.    
    DeviceVector<float> distances(
            res,
            AllocInfo(AllocType::Other, device, MemorySpace::Device, stream));
    distances.resize(n, stream);

    // 'assignments' will store the cluster ID (0 to nlist-1) for each vector.
    DeviceVector<idx_t> assignments(
            res,
            AllocInfo(AllocType::Other, device, MemorySpace::Device, stream));
    assignments.resize(n, stream);

    // B. Perform the Search

    // For each of the n vectors in x, find the nearest neighbor in quantizer.
    // Result: assignments[i] = ID of the cluster centroid closest to x[i].
    this->quantizer->search(n, x, 1, distances.data(), assignments.data());

    // =========================================================
    // 2. Stage 2: Parallel Append
    // =========================================================
    // Now we know where each vector goes. We invoke the custom CUDA kernel
    // to physically copy the data into the SlabManager.

    // A. Get the lightweight Device View
    // This creates the pass-by-value struct containing raw pointers to the Slab
    // pool.
    auto manager_view = slab_manager_->getDeviceView();

    // B. Launch the Custom Kernel (Defined in gpu/impl/SIVFAppend.cuh)
    // This kernel will:
    //  - Read assignments[i] to know the target list.
    //  - Atomic CAS to update list_heads_[assignments[i]].
    //  - Allocate new slabs if a list is full (using
    //  manager_view.allocate_slab).
    //  - Write vector x[i] and id[i] into the slab.
    runSIVFAppend(
            manager_view,
            list_heads_->data(),
            slab_id_buffer_->data(),
            (int)n,
            this->d,
            assignments.data(),
            x,
            ids,
            stream);

    // =========================================================
    // 3. Update Global State
    // =========================================================
    // Manually update the total vector count in the base class.
    // Standard GpuIndexIVF does this internally, but since we overrode
    // addImpl_, we must maintain the book-keeping ourselves.
    this->ntotal += n;
}

/**
 * @brief The core implementation for searching vectors in the SIVF index.
 * * This overrides the standard GpuIndexIVF::searchImpl_.
 * * Workflow:
 * 1. Coarse Search: Use the quantizer to find the 'nprobe' nearest clusters.
 * 2. Fine Search: Invoke custom kernel to scan the linked lists (slabs) of
 * those clusters.
 *
 * @param n Number of query vectors.
 * @param x Pointer to query vectors (Host or Device).
 * @param k Number of nearest neighbors to return (top-k).
 * @param distances Output buffer for distances (size n * k).
 * @param labels Output buffer for IDs (size n * k).
 * @param params Optional search parameters (e.g., to override nprobe).
 */
void GpuIndexSIVF::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {

    // =========================================================
    // 0. Safety Checks
    // =========================================================
    FAISS_THROW_IF_NOT_MSG(is_slab_initialized_, "SIVF not initialized");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "SIVF not trained");

    // =========================================================
    // 1. Parameter Handling (Dynamic Override)
    // =========================================================
    // Default to the index's configured nprobe.
    int nprobe = this->nprobe;
    
    // Check if the user passed runtime parameters to override nprobe.
    if (params) {
        const IVFSearchParameters* ivf_params =
                dynamic_cast<const IVFSearchParameters*>(params);
        if (ivf_params) {
            nprobe = ivf_params->nprobe;
        }
    }

    // =========================================================
    // 2. Coarse Quantization (Stage 1: The Filter)
    // =========================================================
    // We need to find which 'nprobe' clusters are most likely to contain the
    // neighbors. This is done by searching the Quantizer (centroids).

    auto stream = resources_->getDefaultStream(getCurrentDevice());

    // Use Faiss DeviceTensor for RAII memory management on GPU.
    // Dimensions: [n, nprobe]
    // These buffers hold the results of the coarse search.
    DeviceTensor<float, 2, true> coarse_dis(
            resources_.get(),
            makeDevAlloc(AllocType::Other, stream),
            {(int)n, nprobe});
    DeviceTensor<idx_t, 2, true> coarse_ids(
            resources_.get(),
            makeDevAlloc(AllocType::Other, stream),
            {(int)n, nprobe});

    // Invoke the Quantizer (GpuIndexFlat).
    // For each query vector, find the top 'nprobe' nearest centroids.
    // coarse_ids[i] will contain the Cluster IDs (0..nlist-1) to scan.
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_ids.data());

    // ================== [DEBUG START] ==================
    // 1. Check if Quantizer is empty
    // if (quantizer->ntotal == 0) {
    //     printf("[ERROR] Quantizer is EMPTY! (ntotal=0). Did training fail?\n");
    // } else {
        // printf("[DEBUG] Quantizer ntotal = %ld\n", quantizer->ntotal);
    // }
    // ================== [DEBUG END] ====================

    // =========================================================
    // 3. Fine-grained Search (Stage 2: Scan)
    // =========================================================
    // Now we know WHICH lists to look into (from coarse_ids).
    // We invoke our custom kernel to traverse the Slab Linked Lists.

    // Cast away constness.
    // The search method is 'const', but getting the device view might
    // technically touch some internal pointers (though it shouldn't modify
    // data).
    SlabManager* mutable_mgr = const_cast<SlabManager*>(slab_manager_);

    // Get the lightweight struct to pass to the kernel.
    SlabManagerDevice device_view = mutable_mgr->getDeviceView();

    // Launch the custom kernel (defined in gpu/impl/SIVFSearch.cuh).
    // This kernel will:
    //  - Parallelize over queries (n) and probes (nprobe).
    //  - Traverse the linked list starting at list_heads_[coarse_ids[i]].
    //  - Compute distances for all vectors in all slabs in the chain.
    //  - Maintain a Top-K heap for each query.
    runSIVFSearch(
            device_view,
            list_heads_->data(),
            slab_id_buffer_->data(),
            (int)n, // Cast idx_t -> int
            this->d,
            k,
            nprobe,
            x,
            coarse_ids.data(),
            distances,
            labels,
            stream
    );
}

/**
 * @brief Resets the SIVF index to an empty state. This involves:
 * 1. Resetting the quantizer (clearing centroids and assignments).
 * 2. Resetting all list heads to -1 (indicating empty clusters).
 * 3. (TODO) Resetting the SlabManager's internal state (e.g., free list top).
 */
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

/// @brief Not implemented for SIVF. The quantizer is trained as part of the
/// main train() method, and there is no separate sub-quantizer training step.
/// If the user needs to retrain the quantizer, they should call train() again
/// with the new training data. This method is a no-op for SIVF and is provided
/// to satisfy the interface requirements of IndexIVFInterface.
void GpuIndexSIVF::updateQuantizer() {
    // Callback invoked when the user replaces the CPU-side Quantizer.
    // No specific handling required for SIVF currently.
    // TODO: If we later add GPU-side state that depends on the quantizer (e.g.,
    // precomputed norms), we would need to update that state here.
}

/**
 * @brief Trains the index by performing K-Means clustering to find centroids.
 *
 * * Standard Faiss training involves clustering the input data 'x' into 'nlist'
 * clusters. The centroids of these clusters become the "keys" in the Quantizer
 * (GpuIndexFlat).
 * 
 * * This implementation includes a robust fallback:
 * 1. Tries the standard GpuIndexIVF training.
 * 2. If that fails (quantizer remains empty), it manually executes a
 * GPU-accelerated K-Means.
 *
 * @param n Number of training vectors.
 * @param x Pointer to the training vectors (host memory).
 */
void GpuIndexSIVF::train(idx_t n, const float* x) {
    
    // =========================================================
    // 1. Attempt Standard Training
    // =========================================================
    GpuIndexIVF::train(n, x);

    // =========================================================
    // 2. Verification & Fallback Strategy
    // =========================================================
    // Check if the quantizer is actually populated.
    // ntotal == 0 implies the base training failed silently or didn't commit
    // the centroids. This can happen in custom GPU indices if memory states
    // aren't perfectly synced.
    if (this->quantizer->ntotal == 0) {
        // printf("[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...\n");

        // === Optimized Fallback: Manual GPU Accelerated Clustering ===

        // A. Setup Clustering Parameters
        // We need 'nlist' centroids (one for each inverted list).
        faiss::Clustering clus(this->d, this->nlist);
        clus.verbose = false; // Suppress internal logging for cleaner output
        clus.niter = 20; // 20 iterations. High enough for convergence

        // B. GPU-Accelerated Assignment
        // K-Means has two steps: 1. Assign points to nearest centroid. 2.
        // Update centroids. Step 1 is the bottleneck (Nearest Neighbor Search).
        // By passing '*this->quantizer' (which is a GpuIndexFlat), we force
        // Faiss to use the GPU for the Assignment step. If we didn't pass this,
        // Faiss might use a slow CPU index by default.
        this->quantizer->reset();
        clus.train(n, x, *this->quantizer);

        // =========================================================
        // 3. Commit Results
        // =========================================================

        // C. Populate the Quantizer
        // The result of K-Means is stored in 'clus.centroids' (usually in CPU
        // RAM). We must strictly add them into the GPU Quantizer so it can
        // function as an index.
        this->quantizer->reset();

        // Upload the calculated centroids (d * nlist floats) to the GPU index.
        // After this, 'quantizer->ntotal' should equal 'nlist'.
        this->quantizer->add(this->nlist, clus.centroids.data());

        // Mark the flag manually because we bypassed the standard flow.
        this->is_trained = true;

        // printf("[SIVF::train] GPU K-Means complete. Quantizer populated with
        // %ld centroids.\n",
        //        this->quantizer->ntotal);
    }
}

} // namespace gpu
} // namespace faiss
