import time
import os
import gc
import numpy as np
import cupy as cp
import cuvs.neighbors.cagra as cagra_api

# --- Dataset Paths ---
BASE_DIR = os.path.expanduser("~/ElasticIVF/hpdic/data")
DATASETS = {
    "SIFT1M": {
        "path": os.path.join(BASE_DIR, "sift/sift_base.fvecs"),
        "dim": 128,
        "format": "fvecs",
        "n_load": 1_000_000
    },
    "T2I-1B": {
        "path": os.path.join(BASE_DIR, "t2i/t2i_base_1M.fbin"),
        "dim": 200,
        "format": "fbin",
        "n_load": 1_000_000
    },
    "GIST1M": {
        "path": os.path.join(BASE_DIR, "gist/gist_base.fvecs"),
        "dim": 960,
        "format": "fvecs",
        "n_load": 1_000_000
    }
}

# --- IO Helpers (Same as before) ---
def read_fvecs(filename, n_max=None):
    print(f"   Loading fvecs: {filename}...")
    try:
        x = np.fromfile(filename, dtype='float32')
    except FileNotFoundError:
        return None
    with open(filename, 'rb') as f:
        d = np.fromfile(f, dtype='int32', count=1)[0]
    try:
        x = x.reshape(-1, d + 1)
    except ValueError:
        return None
    if n_max is not None and n_max < x.shape[0]:
        x = x[:n_max]
    return x[:, 1:].copy()

def read_fbin(filename, n_max=None):
    print(f"   Loading fbin: {filename}...")
    try:
        with open(filename, 'rb') as f:
            header = np.fromfile(f, dtype='int32', count=2)
            n_points, n_dim = header[0], header[1]
            if n_max is not None and n_max < n_points:
                n_points = n_max
            x = np.fromfile(f, dtype='float32', count=n_points * n_dim)
            x = x.reshape(n_points, n_dim)
        return x
    except FileNotFoundError:
        return None

def normalize_data(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data /= norms
    return data

# --- Benchmark Core ---
def run_benchmark(dataset_name, config):
    print(f"\n========================================")
    print(f"Processing {dataset_name} (Dim: {config['dim']})")
    print(f"========================================")

    # 1. Load Data
    if config['format'] == 'fvecs':
        data = read_fvecs(config['path'], config['n_load'])
    elif config['format'] == 'fbin':
        data = read_fbin(config['path'], config['n_load'])
    else:
        return None, None

    if data is None:
        print(f"Error: Could not load data.")
        return None, None

    # Pre-process
    data = data.astype(np.float32)
    data = normalize_data(data)
    n_total, d = data.shape
    
    # Define Delete Batch Size (10k)
    DELETE_SIZE = 10_000
    
    # 2. Measure Ingestion (Initial Build)
    print(f"-> Measuring Ingestion (Build {n_total})...")
    build_params = cagra_api.IndexParams(
        metric="sqeuclidean",
        intermediate_graph_degree=64,
        graph_degree=32
    )
    
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    index = cagra_api.build(build_params, data)
    cp.cuda.Stream.null.synchronize()
    
    build_time = time.time() - start_time
    ingestion_rate = (n_total / build_time) / 1000.0
    print(f"   Build Time: {build_time:.4f} s")
    print(f"   Ingestion: {ingestion_rate:.2f} K vec/s")

    # 3. Measure TRUE Deletion (Rebuild)
    # To physically remove vectors and reclaim memory, CAGRA must rebuild.
    print(f"-> Measuring TRUE Deletion (Rebuild {n_total - DELETE_SIZE})...")
    
    # Simulate removing 10k vectors from the dataset
    # We keep the first (N - 10k) vectors
    new_size = n_total - DELETE_SIZE
    data_reduced = data[:new_size].copy() # Physical copy to simulate compaction
    
    del index # Delete old index
    del data  # Delete old data
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Rebuild
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    
    index_new = cagra_api.build(build_params, data_reduced)
    
    cp.cuda.Stream.null.synchronize()
    rebuild_time = time.time() - start_time
    rebuild_time_ms = rebuild_time * 1000.0
    
    print(f"   Rebuild Time: {rebuild_time:.4f} s")
    print(f"   True Deletion Latency: {rebuild_time_ms:.2f} ms")
    
    # Cleanup
    del index_new
    del data_reduced
    gc.collect()
    
    return ingestion_rate, rebuild_time_ms

def main():
    results = {}
    
    for name, config in DATASETS.items():
        rate, lat = run_benchmark(name, config)
        if rate is not None:
            results[name] = {"Add": rate, "TrueDel": lat}

    print("\n\n")
    print("="*50)
    print("   CAGRA Benchmark: True Deletion (Rebuild)   ")
    print("="*50)
    print(f"{'Dataset':<10} | {'Ingest (K/s)':<12} | {'True Del (ms)':<12}")
    print("-" * 48)
    for name, metrics in results.items():
        print(f"{name:<10} | {metrics['Add']:<12.2f} | {metrics['TrueDel']:<12.2f}")
    print("="*50)
    print("NOTE: 'True Del' represents the cost of physical memory")
    print("reclamation, which requires a full graph rebuild.")

if __name__ == "__main__":
    main()

# Example Output:
# (myenv) cc@rtx6000:~/ElasticIVF/hpdic/script$ python bench_cagra.py

# ========================================
# Processing SIFT1M (Dim: 128)
# ========================================
#    Loading fvecs: /home/cc/ElasticIVF/hpdic/data/sift/sift_base.fvecs...
# -> Measuring Ingestion (Build 1000000)...
#    Build Time: 3.2717 s
#    Ingestion: 305.65 K vec/s
# -> Measuring TRUE Deletion (Rebuild 990000)...
#    Rebuild Time: 3.0305 s
#    True Deletion Latency: 3030.54 ms

# ========================================
# Processing T2I-1B (Dim: 200)
# ========================================
#    Loading fbin: /home/cc/ElasticIVF/hpdic/data/t2i/t2i_base_1M.fbin...
# -> Measuring Ingestion (Build 1000000)...
#    Build Time: 3.8610 s
#    Ingestion: 259.00 K vec/s
# -> Measuring TRUE Deletion (Rebuild 990000)...
#    Rebuild Time: 3.7734 s
#    True Deletion Latency: 3773.44 ms

# ========================================
# Processing GIST1M (Dim: 960)
# ========================================
#    Loading fvecs: /home/cc/ElasticIVF/hpdic/data/gist/gist_base.fvecs...
# -> Measuring Ingestion (Build 1000000)...
#    Build Time: 10.2942 s
#    Ingestion: 97.14 K vec/s
# -> Measuring TRUE Deletion (Rebuild 990000)...
#    Rebuild Time: 10.2157 s
#    True Deletion Latency: 10215.72 ms



# ==================================================
#    CAGRA Benchmark: True Deletion (Rebuild)   
# ==================================================
# Dataset    | Ingest (K/s) | True Del (ms)
# ------------------------------------------------
# SIFT1M     | 305.65       | 3030.54     
# T2I-1B     | 259.00       | 3773.44     
# GIST1M     | 97.14        | 10215.72    
# ==================================================
# NOTE: 'True Del' represents the cost of physical memory
# reclamation, which requires a full graph rebuild.
# (myenv) cc@rtx6000:~/ElasticIVF/hpdic/script$ 