/**
 * @file faiss/gpu/impl/SIVFSearch.cu
 * @brief Implementation of the SIVF search kernel for GPU.
 * @author Dongfang Zhao (dzhao@uw.edu)
 * @date February 2026
 *
 * @details This file contains the CUDA kernel and host wrapper for executing
 * the search phase of the SIVF index on the GPU. The kernel traverses the
 * slab-based inverted lists, computes distances between query vectors and
 * indexed vectors, and maintains a thread-local top-k heap for each query. The
 * results are then reduced and written back to global memory.
 */

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SIVFSearch.cuh>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Limits.cuh>

namespace faiss {
namespace gpu {

// Maximum K supported by the thread-local sorted array.
// Capped at 32 to guarantee the array fits inside ultra fast hardware
// registers.
constexpr int MAX_K = 32;

/**
 * Helper: Insert a distance and label pair into a thread-local sorted array.
 *
 * @param dists Array of current top-k distances.
 * @param labels Array of current top-k labels.
 * @param k The target number of nearest neighbors to track.
 * @param dist The computed distance of the new vector.
 * @param label The ID of the new vector.
 *
 * @details This function maintains a sorted array of the top-k closest
 * neighbors for a single thread. When a new distance is computed, it checks if
 * it is smaller than the largest distance currently in the array (dists[k -
 * 1]). If it is, the function performs a linear scan from the end of the array
 * to find the correct insertion point, shifting larger distances and their
 * corresponding labels to the right. Finally, it inserts the new distance and
 * label at the correct position to maintain the sorted order. This approach is
 * efficient for small values of k (up to 32) and avoids the overhead of a more
 * complex data structure like a binary heap, which would be unnecessary for
 * such a small fixed size.
 *
 * @note Technically, this is a "sorted array" rather than a heap, because K is
 * small and we want to avoid the overhead of maintaining a binary heap
 * structure. The add_to_heap function performs a linear scan to find the
 * correct insertion point for the new distance, ensuring that the array remains
 * sorted in ascending order. This allows us to efficiently keep track of the
 * top-k closest neighbors for each thread without the complexity of a more
 * traditional heap implementation.
 */
__device__ inline void add_to_heap(
    float* dists,
    idx_t* labels,
    int k,
    float dist,
    idx_t label) {
        
    // Step 1:
    // dists[k - 1] is the longest distance currently in our top-k list.
    // If the new dist is larger than this, it has no chance of entering the
    // top-k. We skip it immediately to save compute cycles.
    if (dist < dists[k - 1]) {
        int i = k - 1;

        // Step 2: Linear Scan and Shift
        // Traverse backwards from the end of the array.
        // As long as the current element is larger than our new dist,
        // we shift it one position to the right to make room.
        while (i > 0 && dist < dists[i - 1]) {
            dists[i] = dists[i - 1];
            labels[i] = labels[i - 1];
            i--;
        }

        // Step 3: Insert
        // The correct sorted position is found. Insert the new values.
        dists[i] = dist;
        labels[i] = label;
    }
}

/**
 * Kernel: SIVF Search
 *
 * Executes the search over inverted lists associated with the query coarse
 * centroids.
 *
 * Execution Mapping:
 * * 1 Query is processed by exactly 1 Block.
 * * 1 Block contains exactly 32 Threads (1 Warp).
 *
 * @param manager Device view of the slab memory manager containing data and
 * metadata.
 * @param list_heads Array of size nlist storing the first slab index for each
 * cluster.
 * @param slab_ids Array mapping physical slot indices to logical vector IDs.
 * @param num_queries Total number of query vectors in this execution batch.
 * @param dim Dimensionality of the vectors.
 * @param k Number of nearest neighbors to retrieve for each query.
 * @param nprobe Number of clusters to search per query.
 * @param queries Flattened array of query vectors. Size is num_queries
 * multiplied by dim.
 * @param coarse_ids Target cluster IDs from coarse quantization. Size is
 * num_queries multiplied by nprobe.
 * @param out_distances Output array for top k distances. Size is num_queries
 * multiplied by k.
 * @param out_labels Output array for top k labels. Size is num_queries
 * multiplied by k.
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

    // ===================================================
    // 1. Load Query into Shared Memory
    // ===================================================

    // Utilize all 32 threads to cooperatively load the query vector.
    // This avoids redundant global memory reads during distance computation.
    __shared__ float shared_query[256];
    for (int i = tid; i < dim; i += blockDim.x)
        shared_query[i] = queries[query_idx * dim + i];

    // Barrier ensures the entire query is loaded before any thread proceeds.
    __syncthreads();

    // ===================================================
    // 2. Initialize Thread-Local Heap
    // ===================================================

    // Allocate max k tracking arrays directly in physical registers.
    float my_dists[MAX_K];
    idx_t my_labels[MAX_K];
    for (int i = 0; i < k; ++i) {
        my_dists[i] = Limits<float>::getMax();
        my_labels[i] = -1;
    }

    // ===================================================
    // 3. Traverse Inverted Lists (Probes)
    // ===================================================

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

            // Use standard struct copy logic.
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

    // ===================================================
    // 4. Reduction: Aggregate Results from all Threads
    // ===================================================

    // Ensure all threads have finished traversing their assigned slots.
    __syncthreads();

    // Allocate shared memory workspace for the reduction phase.
    __shared__ float final_dists[MAX_K * 32];
    __shared__ idx_t final_labels[MAX_K * 32];

    // Every thread flushes its local register heap into the shared workspace.
    if (tid < 32) {
        for (int i = 0; i < k; ++i) {
            final_dists[tid * k + i] = my_dists[i];
            final_labels[tid * k + i] = my_labels[i];
        }
    }

    // Barrier ensures all local heaps are visible in shared memory.
    __syncthreads();

    // Thread 0 assumes responsibility for the final serial merge.
    if (tid == 0) {
        // Iterate through the results submitted by threads 1 through 31.
        // TODO: Warp shuffle-based reduction could be implemented here for
        // better performance, but a simple serial merge is sufficient for
        // correctness and simplicity at this stage.
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
        // ==================================================
        // The following code is experimental and demonstrates how to perform
        // the reduction using warp shuffle
        // ==================================================
        // Tree reduction using register shuffle instructions.
        // Loop over offset: 16, 8, 4, 2, 1
        /**
        for (int offset = 16; offset > 0; offset /= 2) {
            // Iterate through the local heap to fetch each element
            for (int i = 0; i < k; ++i) {
                // Fetch distance and label from the thread 'offset' steps ahead
                float remote_dist =
                    __shfl_down_sync(0xffffffff, my_dists[i], offset);
                idx_t remote_label =
                    __shfl_down_sync(0xffffffff, my_labels[i], offset);

                // Only the lower half of active threads in this step perform
                // the merge This prevents duplicate self insertion from out of
                // bounds shuffle reads
                if (tid < offset && remote_label != -1) {
                    add_to_heap(
                        my_dists, my_labels, k, remote_dist, remote_label);
                }
            }
        }
        */

        // Persist the consolidated top k results back to global memory.
        for (int i = 0; i < k; ++i) {
            out_distances[query_idx * k + i] = my_dists[i];
            out_labels[query_idx * k + i] = my_labels[i];
        }
    }
}

/**
 * Host wrapper function to launch the SIVF search kernel.
 *
 * This function defines the execution configuration (Grid and Block dimensions)
 * and dispatches the search task to the GPU asynchronously via the provided
 * stream.
 *
 * @param manager The lightweight device-side view of the SlabManager.
 * @param list_heads Array of head slab indices for all inverted lists.
 * @param slab_ids Array mapping physical slab slots to logical vector IDs.
 * @param num_queries The number of query vectors in this batch.
 * @param dim The dimensionality of each vector.
 * @param k The number of nearest neighbors to retrieve per query.
 * @param nprobe The number of inverted lists to scan per query.
 * @param queries Device pointer to the flat array of query vectors.
 * @param coarse_ids Device pointer to the array [num_queries * nprobe]
 * containing target cluster IDs.
 * @param out_distances Device pointer to the output array for top-k distances.
 * @param out_labels Device pointer to the output array for top-k labels/IDs.
 * @param stream The CUDA stream on which to enqueue the kernel execution.
 */
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

    // Launch configuration strategy:
    // Grid Size (Blocks): num_queries. Each block independently processes
    // exactly one query vector. Block Size (Threads): 32. This represents
    // exactly one Warp. Rationale: A slab contains 32 slots. By assigning 32
    // threads to a block, the warp can evaluate an entire slab in a single,
    // perfectly coalesced parallel step.
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

    // Safety macro: Captures any asynchronous launch errors (e.g., out of
    // resources, invalid config) that might occur immediately upon pushing the
    // kernel to the queue.
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
