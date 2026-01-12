/**
 * faiss/gpu/GpuIndexSIVF.cu
 */

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h> // [HPDIC]
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/SIVFAppend.cuh> // [HPDIC]
#include <faiss/gpu/impl/SlabManager.cuh>

namespace faiss {
namespace gpu {

// ===========================================================
// 构造与析构
// ===========================================================

GpuIndexSIVF::GpuIndexSIVF(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFConfig config)
        // [注意] 这里的 0.0f 是 metricArg，对应基类构造函数
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          slab_manager_(nullptr),
          is_slab_initialized_(false),
          list_heads_(nullptr) {
    // SIVF 依然需要一个 Coarse Quantizer (聚类中心索引)
    // 我们在这里初始化它，但在 train() 中训练它
    if (!this->quantizer) {
        this->quantizer =
                new GpuIndexFlat(provider, dims, metric, config.flatConfig);
        this->own_fields = true; // 让父类负责释放 quantizer
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

void GpuIndexSIVF::initSlabManager(size_t max_vectors, size_t slab_pool_size) {
    FAISS_THROW_IF_NOT_MSG(
            !is_slab_initialized_, "SlabManager already initialized");

    auto res = resources_.get();
    int device = getCurrentDevice();
    auto stream = res->getDefaultStream(device);

    slab_manager_ =
            new SlabManager(res, device, max_vectors, slab_pool_size, this->d);

    // [新增] 初始化 list_heads_，大小为 nlist，初始值为 -1
    list_heads_ = new DeviceVector<int>(
            res,
            AllocInfo(AllocType::Other, device, MemorySpace::Device, stream));
    list_heads_->resize(this->nlist, stream);

    // 将 list_heads_ 填满 -1 (SIVF_NULL_SLAB)
    // 可以用 thrust::fill 或写个简单 kernel，Faiss 有工具吗？
    // 我们可以简单地 copy 一个全 -1 的 host vector 过去，或者用 cudaMemset
    // (按字节设为0xFF = -1)
    cudaMemsetAsync(list_heads_->data(), -1, this->nlist * sizeof(int), stream);

    is_slab_initialized_ = true;
}

// ===========================================================
// Public Overrides
// ===========================================================

void GpuIndexSIVF::train(idx_t n, const float* x) {
    // 复用基类的 train 逻辑 (主要是训练 quantizer)
    // SIVF 的 Slab 结构本身不需要训练
    GpuIndexIVF::train(n, x);
}

size_t GpuIndexSIVF::remove_ids(const faiss::IDSelector& sel) {
    FAISS_THROW_IF_NOT_MSG(is_slab_initialized_, "SIVF not initialized");

    // TODO: 暂时留空，稍后我们来填这个核心逻辑
    // 这里的返回值应该是被删除的向量数量
    return 0;
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

    // TODO: 暂时留空，等待下一步实现 Slab 搜索
    // printf("SIVF searchImpl_ called\n");
}

} // namespace gpu
} // namespace faiss