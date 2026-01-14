/**
 * faiss/gpu/GpuIndexSIVF.cu
 */

#include <faiss/Clustering.h> // 用于手动 KMeans
#include <faiss/IndexFlat.h>  // 用于 CPU 临时 Index
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h> // [HPDIC]
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SIVFAppend.cuh> // [HPDIC]
#include <faiss/gpu/impl/SIVFSearch.cuh>
#include <faiss/gpu/impl/SlabManager.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh> // 确保包含 DeviceTensor

namespace faiss {
namespace gpu {

// ===========================================================
// 构造与析构
// ===========================================================

// faiss/gpu/GpuIndexSIVF.cu

GpuIndexSIVF::GpuIndexSIVF(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFConfig config)
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          slab_manager_(nullptr),
          is_slab_initialized_(false),
          list_heads_(nullptr) {
    // [关键修正 1] 强制标记为未训练，确保 index.train() 真正执行！
    this->is_trained = false;

    // 初始化 Quantizer (如果没有传进来的话)
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
}

void GpuIndexSIVF::initSlabManager(size_t max_vectors, size_t pool_size) {
    if (is_slab_initialized_)
        return;

    int device = getCurrentDevice();
    auto stream = resources_->getDefaultStream(device);

    // 1. 初始化 SlabManager
    // 这里的 slab_manager_ 构造函数内部也是用的 explicit
    // AllocInfo，所以它没报错
    slab_manager_ = new SlabManager(
            resources_.get(), device, max_vectors, pool_size, this->d);

    // 2. 初始化 List Heads
    if (list_heads_ == nullptr) {
        // [修复] 显式构造 AllocInfo，避免 makeDevAlloc 可能的默认参数问题
        AllocInfo info(AllocType::Other, device, MemorySpace::Device, stream);

        list_heads_ = new DeviceVector<int>(resources_.get(), info);
    }

    // 分配空间
    // 确保 nlist 有效
    FAISS_ASSERT(this->nlist > 0);
    list_heads_->resize(this->nlist, stream);

    // [新增] 检查分配是否成功
    if (list_heads_->data() == nullptr) {
        printf("[ERROR] Failed to allocate list_heads_ (size=%d)\n",
               this->nlist);
        FAISS_THROW_MSG("DeviceVector allocation failed");
    }

    // 初始化为 -1
    // 这里使用 data() 获取指针，确保指针有效
    CUDA_VERIFY(cudaMemsetAsync(
            list_heads_->data(), -1, this->nlist * sizeof(int), stream));

    is_slab_initialized_ = true;
}

// 声明外部定义的启动函数
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
        // 简单处理：对于非 Batch Selector，暂不支持或抛出异常
        FAISS_THROW_MSG(
                "SIVF remove_ids currently ONLY supports IDSelectorBatch");
    }

    int num_removed = 0;
    auto stream = resources_->getDefaultStream(config_.device);

    // 调用我们在 SIVFDeletion.cu 里写好的逻辑
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

// [匹配头文件] addImpl_ (带下划线), 参数 idx_t
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
            list_heads_->data(), // [修改] 使用 ->data()
            (int)n,
            this->d,
            assignments.data(),
            x,
            ids,
            stream);

    // Faiss 的基类不会自动帮你加这个数，得自己加。
    this->ntotal += n;
}

// [匹配头文件] searchImpl_ (带下划线), 参数 k 是 int
void GpuIndexSIVF::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_slab_initialized_, "SIVF not initialized");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "SIVF not trained");

    // 2. 处理 nprobe (支持通过 params 动态传入)
    int nprobe = this->nprobe;
    if (params) {
        const IVFSearchParameters* ivf_params =
                dynamic_cast<const IVFSearchParameters*>(params);
        if (ivf_params) {
            nprobe = ivf_params->nprobe;
        }
    }

    // 3. Coarse Quantization (第一级粗搜)
    // 使用 Faiss 的 DeviceTensor 自动管理显存
    auto stream = resources_->getDefaultStream(getCurrentDevice());

    // 注意：Faiss GPU 内部很多地方用 int 索引，这里强转一下 n (通常 batch
    // 不会超过 20亿)
    DeviceTensor<float, 2, true> coarse_dis(
            resources_.get(),
            makeDevAlloc(AllocType::Other, stream),
            {(int)n, nprobe});
    DeviceTensor<idx_t, 2, true> coarse_ids(
            resources_.get(),
            makeDevAlloc(AllocType::Other, stream),
            {(int)n, nprobe});

    // 调用 Quantizer
    quantizer->search(n, x, nprobe, coarse_dis.data(), coarse_ids.data());

    // ================== [DEBUG START] ==================
    // // 1. 检查 Quantizer 是否为空
    // if (quantizer->ntotal == 0) {
    //     printf("[ERROR] Quantizer is EMPTY! (ntotal=0). Did training fail?\n");
    // } else {
    //     // printf("[DEBUG] Quantizer ntotal = %ld\n", quantizer->ntotal);
    // }

    // // 2. 检查粗搜结果是否全是 -1
    // std::vector<idx_t> host_ids(nprobe);
    // // 从 GPU 拷贝第 0 个 query 的结果到 CPU
    // cudaMemcpyAsync(
    //         host_ids.data(),
    //         coarse_ids.data(),
    //         nprobe * sizeof(idx_t),
    //         cudaMemcpyDeviceToHost,
    //         stream);
    // cudaStreamSynchronize(stream); // 强制同步，确保读到数据

    // if (host_ids[0] == -1) {
    //     printf("[ERROR] Coarse Quantizer returned -1! Printing first %d results:\n",
    //            nprobe);
    //     for (int i = 0; i < nprobe; ++i)
    //         printf("%ld ", host_ids[i]);
    //     printf("\n");
    // }
    // ================== [DEBUG END] ====================

    // 4. Fine-grained Search (调用我们的 Kernel)
    // 去掉 const 限制，因为我们需要获取 DeviceView
    SlabManager* mutable_mgr = const_cast<SlabManager*>(slab_manager_);

    SlabManagerDevice device_view = mutable_mgr->getDeviceView();
    runSIVFSearch(
            device_view,
            list_heads_->data(),
            (int)n, // 强转 idx_t -> int
            this->d,
            k,
            nprobe,
            x,
            coarse_ids
                    .data(), // 只要 ID，不需要 coarse_dis (因为我们还没做残差)
            distances,
            labels,
            stream);
}

void GpuIndexSIVF::reset() {
    // 1. 重置 Quantizer
    if (quantizer) {
        quantizer->reset();
    }

    // 2. 重置链表头 (全部设为 -1)
    if (is_slab_initialized_ && list_heads_) {
        int device = getCurrentDevice();
        auto stream = resources_->getDefaultStream(device);
        cudaMemsetAsync(
                list_heads_->data(), -1, this->nlist * sizeof(int), stream);
    }

    // TODO: SlabManager 也应该 reset (重置 free_list_top)，暂时跳过
    // 下次优化时我们可以在 SlabManager 里加一个 reset() 方法
}

void GpuIndexSIVF::updateQuantizer() {
    // 这是一个回调函数，当用户在 CPU 侧替换了 Quantizer 时会被调用
    // SIVF 暂时不需要特殊处理
}

void GpuIndexSIVF::train(idx_t n, const float* x) {
    // 1. 尝试基类训练
    GpuIndexIVF::train(n, x);

    // 2. 检查是否训练成功
    if (this->quantizer->ntotal == 0) {
        printf("[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...\n");

        // === 优化版保底方案：使用 GPU 加速聚类 ===

        // 1. 设置聚类参数
        faiss::Clustering clus(this->d, this->nlist);
        clus.verbose = true;
        clus.niter = 20; // GPU 很快，可以多跑几轮保证质量

        // 2. [关键] 直接把 GPU Quantizer 传进去！
        // Faiss 会自动利用这个 GPU Index 加速 K-Means 的“分配”阶段
        // 这里的 *this->quantizer 是 GpuIndexFlat，天生支持 GPU Search
        this->quantizer->reset();
        clus.train(n, x, *this->quantizer);

        // 3. 将最终的 centroids 写入 Quantizer
        // 注意：clus.train 会把中间结果存在 clus.centroids (CPU)
        // 我们需要把最终结果再 add 一次进 GPU quantizer
        this->quantizer->reset();
        this->quantizer->add(this->nlist, clus.centroids.data());

        this->is_trained = true;

        printf("[SIVF::train] GPU K-Means complete. Quantizer populated with %ld centroids.\n",
               this->quantizer->ntotal);
    }
}

} // namespace gpu
} // namespace faiss