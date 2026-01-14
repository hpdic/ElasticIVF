#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Limits.cuh>

namespace faiss {
namespace gpu {

__device__ inline void add_to_heap(
        float* dists,
        idx_t* labels,
        int k,
        float dist,
        idx_t label) {
    if (dist < dists[k - 1]) {
        int i = k - 1;
        while (i > 0 && dist < dists[i - 1]) {
            dists[i] = dists[i - 1];
            labels[i] = labels[i - 1];
            i--;
        }
        dists[i] = dist;
        labels[i] = label;
    }
}

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

    __shared__ float shared_query[256];
    for (int i = tid; i < dim; i += blockDim.x)
        shared_query[i] = queries[query_idx * dim + i];
    __syncthreads();

    const int MAX_K = 32;
    float my_dists[MAX_K];
    idx_t my_labels[MAX_K];
    for (int i = 0; i < k; ++i) {
        my_dists[i] = Limits<float>::getMax();
        my_labels[i] = -1;
    }

    for (int p = 0; p < nprobe; ++p) {
        idx_t cluster_id = coarse_ids[query_idx * nprobe + p];
        if (cluster_id == -1)
            continue;

        volatile int* heads_ptr = list_heads;
        int cur_slab = heads_ptr[cluster_id];

        int loop_safety = 0;
        while (cur_slab != -1 && loop_safety < 10000) {
            loop_safety++;

            // [Critical Fix] Use standard struct copy. Do not use int* casting.
            SlabMetadata md = manager.slab_metadata[cur_slab];

            // Safety break against self-loops
            if (md.next_slab_idx == cur_slab)
                break;

            if (tid < 32) {
                if ((md.validity_bitmap >> tid) & 1) {
                    float dist = 0.0f;
                    float* vec_data = manager.slab_data +
                            (size_t)cur_slab * 32 * dim + tid * dim;

                    for (int d = 0; d < dim; ++d) {
                        float diff = shared_query[d] - vec_data[d];
                        dist += diff * diff;
                    }

                    size_t physical_id_idx = (size_t)cur_slab * 32 + tid;
                    idx_t real_id = slab_ids[physical_id_idx];
                    add_to_heap(my_dists, my_labels, k, dist, real_id);
                }
            }
            cur_slab = md.next_slab_idx;
        }
    }

    __syncthreads();
    __shared__ float final_dists[MAX_K * 32];
    __shared__ idx_t final_labels[MAX_K * 32];

    if (tid < 32) {
        for (int i = 0; i < k; ++i) {
            final_dists[tid * k + i] = my_dists[i];
            final_labels[tid * k + i] = my_labels[i];
        }
    }
    __syncthreads();

    if (tid == 0) {
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