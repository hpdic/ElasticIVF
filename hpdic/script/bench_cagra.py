import time
import numpy as np
import cupy as cp
import cuvs.neighbors.cagra as cagra_api
import gc

class SoftDeletionCAGRA:
    def __init__(self, index, n_vectors):
        self.index = index
        self.n_vectors = n_vectors
        # 1. Maintain a Bitmask on GPU (True = Alive, False = Deleted)
        self.alive_mask = cp.ones((n_vectors,), dtype=cp.bool_)
        self.deleted_count = 0

    def delete(self, ids):
        """
        Logical Deletion: Only updates the Bitmask, does not modify graph structure.
        Complexity: O(1) regarding graph size.
        """
        ids_gpu = cp.asarray(ids, dtype=cp.int64)
        self.alive_mask[ids_gpu] = False
        self.deleted_count += len(ids)

    def search(self, queries, k, search_params):
        """
        Filtered Search: Over-fetch -> Filter -> TopK
        """
        # 2. Over-fetching: Search for 2*k neighbors to ensure we have enough after filtering
        fetch_k = k * 2 
        
        # Execute native search
        # Note: cuvs returns a lightweight 'device_ndarray' wrapper, not a standard cupy array
        distances_raw, neighbors_raw = cagra_api.search(search_params, self.index, queries, fetch_k)
        
        # --- FIX: Explicitly convert to CuPy array to enable .copy() and advanced indexing ---
        distances = cp.asarray(distances_raw)
        neighbors = cp.asarray(neighbors_raw)
        
        # 3. Client-side Filtering (Parallel execution on GPU)
        # Check if neighbors are alive: (n_queries, fetch_k) boolean matrix
        is_alive = self.alive_mask[neighbors]
        
        # Set distances of deleted nodes to infinity for sorting
        # CAGRA returns squared L2 distance (larger is further)
        distances_filtered = distances.copy()
        distances_filtered[~is_alive] = cp.inf
        
        # Re-sort to get the true Top-K
        # argsort performs parallel sorting per row on GPU
        sorted_indices = cp.argsort(distances_filtered, axis=1)
        
        # Extract final indices
        final_indices = sorted_indices[:, :k]
        
        # Use advanced indexing to retrieve IDs and Distances from original arrays
        row_ids = cp.arange(queries.shape[0])[:, None]
        final_neighbors = neighbors[row_ids, final_indices]
        final_distances = distances[row_ids, final_indices]
        
        return final_distances, final_neighbors

def main():
    # Configuration
    N_VECTORS = 1_000_000
    D = 128
    N_QUERIES = 10_000
    TOP_K = 10
    DELETE_BATCH = 10_000 # Simulate deleting 10k vectors

    print(f"--- Benchmarking CAGRA with Soft Deletion ---")

    # 1. Data Preparation
    print("Generating data...")
    dataset = np.random.rand(N_VECTORS, D).astype(np.float32)
    queries_host = np.random.rand(N_QUERIES, D).astype(np.float32)
    
    # Randomly select IDs to delete
    delete_ids = np.random.choice(N_VECTORS, DELETE_BATCH, replace=False)

    # 2. Build Index
    print("Building Index...")
    build_params = cagra_api.IndexParams(metric="sqeuclidean")
    start_time = time.time()
    index = cagra_api.build(build_params, dataset)
    print(f"Build finished in {time.time() - start_time:.4f} s")

    # Initialize Soft Deletion Wrapper
    cagra_wrapper = SoftDeletionCAGRA(index, N_VECTORS)

    # 3. Execution Deletion
    print(f"Simulating deletion of {DELETE_BATCH} vectors...")
    
    # Warmup for deletion (to avoid measuring compilation/initialization overhead)
    cp.cuda.Stream.null.synchronize()
    
    start_time = time.time()
    cagra_wrapper.delete(delete_ids)
    cp.cuda.Stream.null.synchronize() # Ensure kernel completion
    delete_time = (time.time() - start_time) * 1000 # ms
    
    print(f"Deletion Latency (Bitmap Update): {delete_time:.4f} ms") 

    # 4. Search Test (Standard vs Soft-Delete)
    print("\n--- Performance Comparison ---")
    queries_gpu = cp.asarray(queries_host)
    search_params = cagra_api.SearchParams()

    # Baseline: Native Search (Ignoring deleted items)
    print("1. Native Search (Ignoring deleted items)...")
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    cagra_api.search(search_params, index, queries_gpu, TOP_K)
    cp.cuda.Stream.null.synchronize()
    native_time = time.time() - start_time
    print(f"   Native QPS: {N_QUERIES / native_time:.2f}")

    # Ours: Filtered Search
    print("2. Filtered Search (Over-fetch + Masking)...")
    cp.cuda.Stream.null.synchronize()
    start_time = time.time()
    cagra_wrapper.search(queries_gpu, TOP_K, search_params)
    cp.cuda.Stream.null.synchronize()
    filtered_time = time.time() - start_time
    print(f"   Filtered QPS: {N_QUERIES / filtered_time:.2f}")

    # Calculate Overhead
    overhead = (filtered_time - native_time) / native_time * 100
    print(f"\nConclusion: Soft Deletion incurred {overhead:.2f}% latency overhead.")
    print("Note: This overhead grows as deletion ratio increases (requires larger over-fetch).")

    # Clean up to prevent Python exit errors
    del index
    del cagra_wrapper
    gc.collect()

if __name__ == "__main__":
    main()

# Example output:
# (myenv) cc@rtx6000:~/ElasticIVF/hpdic/script$ python bench_cagra.py
# --- Benchmarking CAGRA with Soft Deletion ---
# Generating data...
# Building Index...
# Build finished in 6.3406 s
# Simulating deletion of 10000 vectors...
# Deletion Latency (Bitmap Update): 2.4805 ms

# --- Performance Comparison ---
# 1. Native Search (Ignoring deleted items)...
#    Native QPS: 47817.35
# 2. Filtered Search (Over-fetch + Masking)...
#    Filtered QPS: 9468.07

# Conclusion: Soft Deletion incurred 405.04% latency overhead.
# Note: This overhead grows as deletion ratio increases (requires larger over-fetch).
# (myenv) cc@rtx6000:~/ElasticIVF/hpdic/script$ cd ElasticIVF/hpdic/script/










