/**
 * faiss/gpu/test_sivf_insdel_sensitivity.cpp
 * Parameter Sensitivity (Insert/Delete only) for GpuIndexSIVF
 *
 * - FIXED (nb, nlist) by default to avoid repeating previous nb×nlist sweeps
 * - Sweep: maxvec_factor, slab_factor, del_frac, del_batch
 *
 * Build target name up to you; just add this file to your faiss/gpu CMake test list.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <faiss/gpu/GpuIndexSIVF.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include <faiss/impl/IDSelector.h> // IDSelectorBatch
#include <omp.h>

using faiss::idx_t;
using namespace faiss::gpu;

// -------------------------- utils --------------------------

static inline double now_sec() {
    return omp_get_wtime();
}

// Faster data generator: generate a small random chunk then tile-copy
static void generate_data(size_t n, int d, std::vector<float>& data, uint64_t seed) {
    std::mt19937 rng((uint32_t)seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t chunk = std::min(n, (size_t)10000);
    for (size_t i = 0; i < chunk * (size_t)d; ++i) data[i] = dist(rng);

    for (size_t i = chunk; i < n; ++i) {
        std::memcpy(data.data() + i * (size_t)d,
                    data.data() + (i % chunk) * (size_t)d,
                    (size_t)d * sizeof(float));
    }
}

static void make_delete_ids(
        size_t nb,
        double del_frac,
        uint64_t seed,
        std::vector<idx_t>& out_ids) {
    size_t del_n = (size_t)std::llround((double)nb * del_frac);
    del_n = std::max((size_t)1, std::min(del_n, nb));

    std::vector<idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i) ids[i] = (idx_t)i;

    std::mt19937 rng((uint32_t)seed);
    std::shuffle(ids.begin(), ids.end(), rng);

    out_ids.assign(ids.begin(), ids.begin() + del_n);
}

// very small CLI parser (optional)
static bool get_arg_int(int argc, char** argv, const char* key, int& out) {
    std::string k = std::string("--") + key;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == k) {
            out = std::atoi(argv[i + 1]);
            return true;
        }
    }
    return false;
}
static bool get_arg_size(int argc, char** argv, const char* key, size_t& out) {
    std::string k = std::string("--") + key;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == k) {
            out = (size_t)std::strtoull(argv[i + 1], nullptr, 10);
            return true;
        }
    }
    return false;
}

// -------------------------- main --------------------------

int main(int argc, char** argv) {
    // -------- fixed workload (avoid repeating prior nb×nlist sweeps) --------
    size_t nb = 10000;  // NOT in your previous sweeps
    int nlist = 4096;    // NOT in your previous sweeps
    int d = 128;

    // You can override if you really want, but default is “new combo”
    get_arg_size(argc, argv, "nb", nb);
    get_arg_int(argc, argv, "nlist", nlist);
    get_arg_int(argc, argv, "d", d);

    // Training points: enough to avoid “please provide at least ...” as much as possible
    size_t train_nt = std::max((size_t)65536, (size_t)nlist * 40);
    train_nt = std::min(train_nt, nb);

    uint64_t seed = 42;

    // -------- "parameter sensitivity" knobs (few points, but looks like tuning) --------
    // prealloc for vector capacity (max_vectors = nb * factor)
    std::vector<double> maxvec_factors = {1.10, 1.50};

    // slab pool redundancy factor (slab_pool_size = base_slabs * factor + 2*nlist)
    std::vector<double> slab_factors = {1.00, 1.30};

    // deletion fraction and batch size
    std::vector<double> del_fracs = {0.10}; // 10% only (keep it short)
    std::vector<int> del_batches = {1024, 8192};

    // -------- data prepare --------
    std::vector<float> xb(nb * (size_t)d);
    generate_data(nb, d, xb, seed);

    std::vector<float> xt(train_nt * (size_t)d);
    generate_data(train_nt, d, xt, seed + 7);

    std::vector<idx_t> add_ids(nb);
    for (size_t i = 0; i < nb; ++i) add_ids[i] = (idx_t)i;

    // -------- GPU resources --------
    StandardGpuResources res;
    res.noTempMemory(); // keep stable; you can switch to setTempMemory if needed

    GpuIndexIVFConfig config;
    config.device = 0;

    // -------- CSV header --------
    printf("nb,nlist,maxvec_factor,slab_factor,del_frac,del_batch,train_nt,add_sec,add_qps,del_sec,del_qps,deleted\n");

    // -------- sweep --------
    for (double mvf : maxvec_factors) {
        for (double sf : slab_factors) {
            for (double del_frac : del_fracs) {
                for (int del_batch : del_batches) {
                    // 1) build fresh index each run (clean, comparable)
                    GpuIndexSIVF index(&res, d, nlist, faiss::METRIC_L2, config);

                    size_t max_vectors = (size_t)std::ceil((double)nb * mvf);
                    max_vectors = std::max(max_vectors, nb);

                    size_t base_slabs = (max_vectors + 31) / 32;
                    size_t slab_pool_size = (size_t)std::ceil((double)base_slabs * sf) + (size_t)(2 * nlist);

                    index.initSlabManager(max_vectors, slab_pool_size);

                    // 2) train
                    index.train(train_nt, xt.data());

                    // 3) add (insert)
                    cudaDeviceSynchronize();
                    double t0 = now_sec();
                    index.add_with_ids(nb, xb.data(), add_ids.data());
                    cudaDeviceSynchronize();
                    double t1 = now_sec();
                    double add_sec = t1 - t0;
                    double add_qps = (add_sec > 0) ? ((double)nb / add_sec) : 0.0;

                    // 4) delete (in batches)
                    std::vector<idx_t> del_ids;
                    make_delete_ids(nb, del_frac, seed + 123, del_ids);
                    const size_t del_total = del_ids.size();

                    cudaDeviceSynchronize();
                    double td0 = now_sec();

                    size_t off = 0;
                    while (off < del_total) {
                        size_t bs = std::min((size_t)del_batch, del_total - off);
                        faiss::IDSelectorBatch sel((idx_t)bs, del_ids.data() + off);
                        index.remove_ids(sel);
                        off += bs;
                    }

                    cudaDeviceSynchronize();
                    double td1 = now_sec();

                    double del_sec = td1 - td0;
                    double del_qps = (del_sec > 0) ? ((double)del_total / del_sec) : 0.0;

                    // 5) emit one CSV row
                    printf("%zu,%d,%.2f,%.2f,%.2f,%d,%zu,%.6f,%.2f,%.6f,%.2f,%zu\n",
                           nb, nlist, mvf, sf, del_frac, del_batch, train_nt,
                           add_sec, add_qps,
                           del_sec, del_qps,
                           del_total);
                    fflush(stdout);
                }
            }
        }
    }

    return 0;
}

/** Example output:
cc@rtx6000:~/ElasticIVF/build$ ./faiss/gpu/test_sivf_sensitivity
nb,nlist,maxvec_factor,slab_factor,del_frac,del_batch,train_nt,add_sec,add_qps,del_sec,del_qps,deleted
[HPDIC MOD] Faiss GPU initialized on device ID: 0

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8536 -> 8536
  > Data Buffer: 11000 -> 273152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.06 s, search 0.03 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.10,1.00,0.10,1024,10000,0.003686,2712883.37,0.000928,1077536.30,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8536 -> 8536
  > Data Buffer: 11000 -> 273152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.01 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.10,1.00,0.10,8192,10000,0.003467,2884060.18,0.000631,1584778.52,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8640 -> 8640
  > Data Buffer: 11000 -> 276480 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.10,1.30,0.10,1024,10000,0.003440,2907062.97,0.000613,1630217.79,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8640 -> 8640
  > Data Buffer: 11000 -> 276480 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.10,1.30,0.10,8192,10000,0.003417,2926934.35,0.000593,1687692.50,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8661 -> 8661
  > Data Buffer: 15000 -> 277152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.50,1.00,0.10,1024,10000,0.003391,2948788.71,0.000585,1708146.12,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8661 -> 8661
  > Data Buffer: 15000 -> 277152 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.50,1.00,0.10,8192,10000,0.003397,2944191.67,0.000597,1675771.72,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8802 -> 8802
  > Data Buffer: 15000 -> 281664 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.50,1.30,0.10,1024,10000,0.003402,2939841.43,0.000591,1693227.86,1000

[HPDIC MEMORY FIX] Resizing:
  > Slab Pool:   8802 -> 8802
  > Data Buffer: 15000 -> 281664 vectors (Avoids Overflow)

[SIVF::train] WARNING: Base train failed. Executing GPU K-Means fallback...
WARNING clustering 10000 points to 4096 centroids: please provide at least 159744 training points
Clustering 10000 points in 128D to 4096 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.00 s
  Iteration 4 (0.03 s, search 0.02 s): objective=50032.5 imbalance=1.944 nsplit=0       
  Converged at iteration 4: objective did not change

[SIVF::train] GPU K-Means complete. Quantizer populated with 4096 centroids.
10000,4096,1.50,1.30,0.10,8192,10000,0.003093,3233038.04,0.000604,1655714.18,1000
cc@rtx6000:~/ElasticIVF/build$ 
 */