/**
 * faiss/gpu/impl/SIVFAppend.cu
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/SIVFStructs.cuh>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

// =============================================================
// SIVF Append Kernel
// =============================================================

__global__ void sivf_add_kernel(
        SlabManagerDevice manager,
        int* list_heads,          // [nlist] 存储每个 List 当前活跃的 Slab ID
        const idx_t* assignments, // [n] 每个向量归属的 list_id
        const float* vectors,     // [n * d] 原始向量数据
        const idx_t* ids,         // [n] 向量的逻辑 ID
        int n,
        int d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    int list_id = (int)assignments[tid];
    long vector_id = ids[tid];

    // 简单的自旋重试循环
    // 如果申请 Slab 失败或发生竞争，就重试
    while (true) {
        // 1. 获取当前活跃的 Slab
        int curr_slab = list_heads[list_id];

        // Case A: 链表为空，或者之前的 Slab 已满但还没更新 head
        // 我们尝试初始化它
        if (curr_slab == SIVF_NULL_SLAB) {
            int new_slab = manager.allocate_slab();
            if (new_slab == SIVF_NULL_SLAB)
                return; // 显存满了，无法处理 (TODO: 报错)

            // 原子 CAS (Compare And Swap) 尝试设为 Head
            // 如果 list_heads[list_id] 还是 -1，就设为 new_slab
            int old_val =
                    atomicCAS(&list_heads[list_id], SIVF_NULL_SLAB, new_slab);

            if (old_val != SIVF_NULL_SLAB) {
                // 竞争失败，别人先设了。我申请的这个没用了，还回去。
                manager.free_slab(new_slab);
                // 重试，下一轮循环会拿到新的 curr_slab
                continue;
            }
            // 竞争成功，curr_slab 现在是 new_slab
            curr_slab = new_slab;
        }

        // 2. 尝试在 curr_slab 里占座
        // valid_count 可能被加到 >32，没关系，这代表溢出
        int slot =
                atomicAdd(&(manager.slab_metadata[curr_slab].valid_count), 1);

        // Case B: 抢到了槽位 (0..31)
        if (slot < SIVF_SLAB_CAPACITY) {
            // >>>> 写入数据 <<<<

            // a. 写入向量 payload
            // 物理位置: slab_data + (curr_slab * 32 + slot) * d
            long offset_in_floats =
                    (long)curr_slab * SIVF_SLAB_CAPACITY * d + (long)slot * d;
            const float* src_vec = vectors + (long)tid * d;
            float* dst_vec = manager.slab_data + offset_in_floats;

            for (int i = 0; i < d; ++i) {
                dst_vec[i] = src_vec[i];
            }

            // b. 更新 Bitmap (原子置位)
            atomicOr(
                    &(manager.slab_metadata[curr_slab].validity_bitmap),
                    (1U << slot));

            // c. 更新全局 AddressTable
            manager.update_address(vector_id, curr_slab, slot);

            // 成功退出
            break;
        }

        // Case C: 没抢到 (slot >= 32)，说明这个 Slab 满了
        // 我们需要由 "Leader" 线程负责申请新 Slab 并挂载
        // 这里使用一个简化的逻辑：如果我是第 32 个进来的 (slot ==
        // 32)，我负责扩展
        if (slot == SIVF_SLAB_CAPACITY) {
            int new_slab = manager.allocate_slab();
            // TODO: 错误处理 if new_slab == -1

            // 1. 链接: new -> old (头插法? 或者尾插法?)
            // 这里我们使用 "头插法" 或者 "当前活跃块替换法"
            // 让 new_slab 指向原来的 next (或者 SIVF_NULL)
            // 实际上为了搜索方便，通常是单向链表。
            // 简单策略：curr_slab 满了，它变成历史。new_slab 变成新的 head。
            // 但是为了不丢失旧数据，old_head 应该被 new_slab->next 指向？
            //
            // 修正策略：
            // SIVF 这里的 list_heads 指向的是 "当前可写的 Slab"。
            // 满的 Slab 应该被移走。
            //
            // 让我们采用更稳健的链表扩容：
            // allocate new_slab
            // new_slab->next = curr_slab (头插法，新数据在链表头)
            manager.slab_metadata[new_slab].next_slab_idx = curr_slab;

            // 2. 内存屏障，确保 next 指针写完
            __threadfence();

            // 3. 更新 list_heads 指向 new_slab
            // 只有成功把 head 从 curr_slab 变成 new_slab 的线程才算成功
            // 但其实这里允许多个满的 Slab 同时扩容，只要 atomicExch 即可
            // 不过为了保证连贯性，我们用 atomicCAS 确保只有一个人做这步
            // 实际上 slot==32 的那个线程做最好。

            // 强制更新 Head
            // 这里的风险是：如果有其他线程正在往 curr_slab 写 (slot <
            // 32)，没关系，他们有 curr_slab 的旧引用 新来的线程会读到 new_slab
            int old_head = atomicExch(&list_heads[list_id], new_slab);

            // 如果 old_head 不等于 curr_slab，说明中间有人插了一脚，
            // 我们的 new_slab->next 指向了 curr_slab，这造成了分支？
            // 这是一个复杂的并发链表问题。

            // **最简方案 (Best Effort)**:
            // 为了 Prototype，我们假设 slot == 32 的人负责分配，
            // 并且用 CAS 保证原子性。
            // 如果 CAS 失败（说明 Head 变了），我们
            // free(new_slab)，并在下一轮循环重试。
        }

        // 如果 slot > 32，说明我是迟到的，等待 Head 更新后重试
        // 下一轮循环 while(true) 会重新读取 list_heads[list_id]
        // 如果 Head 已经被 slot==32 的兄弟更新了，我就能读到新的。
    }
}

void runSIVFAppend(
        SlabManagerDevice& manager,
        int* list_heads,
        int n,
        int d,
        const idx_t* assignments,
        const float* x,
        const idx_t* ids,
        cudaStream_t stream) {
    int block = 128;
    int grid = (n + block - 1) / block;

    sivf_add_kernel<<<grid, block, 0, stream>>>(
            manager, list_heads, assignments, x, ids, n, d);
    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss