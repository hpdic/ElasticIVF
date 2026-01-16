/**
 * faiss/gpu/impl/SIVFSearch.cu
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Implementation of the exact search kernel for SIVF.
 * This file handles the traversal of slab-based inverted lists, distance computation,
 * and the thread-local top-k heap management within the GPU.
 */

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/impl/SIVFSearch.cuh>

namespace faiss {
namespace gpu {

// Maximum K supported by the on-register heap
constexpr int MAX_K = 32;

/**
 * Helper: Insert a distance/label pair into a sorted heap (priority queue).
 * Maintains the smallest k elements in ascending order (or largest k, depending on metric).
 * Assumes a linear scan insertion for small k (k <= 32).
 */
__device__ inline void add_to_heap(
        float* dists,
        idx_t* labels,
        int k,
        float dist,
        idx_t label) {
    if (dist < dists[k - 1]) {
        int i = k - 1;
        // Shift elements to make room
        while (i > 0 && dist < dists[i - 1]) {
            dists[i] = dists[i - 1];
            labels[i] = labels[i - 1];
            i--;
        }
        // Insert new element
        dists[i] = dist;
        labels[i] = label;
    }
}

/**
 * Kernel: SIVF Search
 *
 * Executes the search over inverted lists associated with the query's coarse centroids.
 * Each block handles one query, and threads within the block (warp-sized) cooperate
 * to scan vectors in parallel.
 *
 * Configuration:
 * - Block Dimension: 32 (1 Warp)
 * - Grid Dimension: num_queries
 */
__global__ void sivf_search_kernel(
        SlabManagerDevice manager,
        int* list_heads,
        idx_t* slab_ids,
        int num_queries,
        int dim,
        int k,
        int nprobe,
        const float* queries,
        const idx_t* coarse_ids,
        float* out_distances,
        idx_t* out_labels) {
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (query_idx >= num_queries)
        return;

    // 1. Load Query into Shared Memory
    // Although dim usually > 32, the loop ensures full loading.
    __shared__ float shared_query[256];
    for (int i = tid; i < dim; i += blockDim.x)
        shared_query[i] = queries[query_idx * dim + i];
    __syncthreads();

    // 2. Initialize Thread-Local Heap
    float my_dists[MAX_K];
    idx_t my_labels[MAX_K];
    for (int i = 0; i < k; ++i) {
        my_dists[i] = Limits<float>::getMax();
        my_labels[i] = -1;
    }

    // 3. Traverse Inverted Lists (Probes)
    for (int p = 0; p < nprobe; ++p) {
        idx_t cluster_id = coarse_ids[query_idx * nprobe + p];
        if (cluster_id == -1)
            continue;

        // Retrieve head slab index
        volatile int* heads_ptr = list_heads;
        int cur_slab = heads_ptr[cluster_id];

        int loop_safety = 0;
        
        // Traverse the linked slabs
        while (cur_slab != -1 && loop_safety < 10000) {
            loop_safety++;

            // [Critical Fix] Use standard struct copy logic.
            // Avoid unsafe casting (e.g., int*) which may violate alignment 
            // or strict aliasing rules.
            SlabMetadata md = manager.slab_metadata[cur_slab];

            // Safety break: Detect and break infinite self-loops
            if (md.next_slab_idx == cur_slab)
                break;

            // Process vectors in the current slab
            // Each thread in the warp handles one slot (0-31)
            if (tid < 32) {
                // Check if the slot contains valid data
                if ((md.validity_bitmap >> tid) & 1) {
                    float dist = 0.0f;
                    float* vec_data = manager.slab_data +
                            (size_t)cur_slab * 32 * dim + tid * dim;

                    // Compute L2 Distance
                    for (int d = 0; d < dim; ++d) {
                        float diff = shared_query[d] - vec_data[d];
                        dist += diff * diff;
                    }

                    // Retrieve global vector ID
                    size_t physical_id_idx = (size_t)cur_slab * 32 + tid;
                    idx_t real_id = slab_ids[physical_id_idx];
                    
                    // Update local heap
                    add_to_heap(my_dists, my_labels, k, dist, real_id);
                }
            }
            // Move to the next slab in the chain
            cur_slab = md.next_slab_idx;
        }
    }

    // 4. Reduction: Aggregate Results from all Threads
    __syncthreads();
    
    // Allocate shared memory for reduction (32 threads * 32 K)
    __shared__ float final_dists[MAX_K * 32];
    __shared__ idx_t final_labels[MAX_K * 32];

    // Dump local heaps to shared memory
    if (tid < 32) {
        for (int i = 0; i < k; ++i) {
            final_dists[tid * k + i] = my_dists[i];
            final_labels[tid * k + i] = my_labels[i];
        }
    }
    __syncthreads();

    // Perform final reduction on Thread 0
    // Note: This is a serial reduction for simplicity. Can be optimized 
    // with bitonic sort or tree reduction for larger K.
    if (tid == 0) {
        // Merge heaps from threads 1..31 into thread 0's heap
        for (int t = 1; t < 32; ++t) {
            for (int i = 0; i < k; ++i) {
                if (final_labels[t * k + i] != -1) {
                    add_to_heap(
                            my_dists,
                            my_labels,
                            k,
                            final_dists[t * k + i],
                            final_labels[t * k + i]);
                }
            }
        }
        // Write final Top-K to global output
        for (int i = 0; i < k; ++i) {
            out_distances[query_idx * k + i] = my_dists[i];
            out_labels[query_idx * k + i] = my_labels[i];
        }
    }
}

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
        cudaStream_t stream) {
    // Launch configuration:
    // One block per query, 32 threads (1 warp) per block.
    // This maps well to the slab width (32 slots).
    sivf_search_kernel<<<num_queries, 32, 0, stream>>>(
            manager,
            list_heads,
            slab_ids,
            num_queries,
            dim,
            k,
            nprobe,
            queries,
            coarse_ids,
            out_distances,
            out_labels);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss