#!/bin/bash

# =============================================================================
# Script Name: search_gist.sh
# Description: Automated grid search script for SIVF on the GIST dataset.
#              It iterates through specified nlist and nprobe parameters
#              to benchmark QPS and Recall performance against Baselines.
# =============================================================================

# ================= Configuration =================
EXEC="$HOME/ElasticIVF/build/test_sivf_gist_search"
LOG_FILE="$HOME/ElasticIVF/hpdic/result/grid_search_gist"

# 1. Training Data Size (Keep large enough to ensure clustering quality)
NT_TRAIN=1000000

# 2. Temporary GPU Memory (1GB is sufficient for GIST execution)
TEMP_MEM=1024

# ================= Grid Search Parameters =================

# Variable A: nlist (Number of cluster centroids)
# SIVF is extremely sensitive to this parameter.
# - 1024: Traditional setting, but clusters are large (~1000 vectors/cluster).
#         Slab linked lists are long, resulting in heavy pointer chasing.
# - 2048/4096: Smaller clusters, shorter Slab linked lists.
#              SIVF's non-contiguous memory layout might perform better here!
NLISTS=(8192)

# Variable B: nprobe (Number of clusters to probe)
# We focus on small probes (1-16) to target high QPS,
# but also retain 32/64 to observe the recall upper bound.
PROBES=(16 32 64 128)

# ================= Execution =================

# Initialize Log
if [ ! -f $LOG_FILE ]; then
    echo "Timestamp, Method, nlist, nprobe, QPS, Recall" > $LOG_FILE
fi

echo "Starting Grid Search..."
echo "Fixed Params: nt_train=$NT_TRAIN, temp_mem=${TEMP_MEM}MB"
echo "Sweeping: nlist={${NLISTS[*]}}, nprobe={${PROBES[*]}}"
echo "--------------------------------------------------------"

for nlist in "${NLISTS[@]}"; do
    echo "========================================"
    echo " Testing nlist = $nlist "
    echo "========================================"
    
    for nprobe in "${PROBES[@]}"; do
        echo -n "[Running] nlist=$nlist, nprobe=$nprobe ... "
        
        # Capture output
        OUTPUT=$($EXEC $nlist $nprobe $NT_TRAIN $TEMP_MEM)
        
        # Print simplistic result to screen (extract lines with QPS)
        echo ""
        echo "$OUTPUT" | grep -E "\[SIVF\]|\[Baseline\]"
        
        # Log to file
        while IFS= read -r line; do
            if [[ "$line" == *"[SIVF]"* ]] || [[ "$line" == *"[Baseline]"* ]]; then
                # Clean format: remove pipes, make comma-separated
                # From: [SIVF] nprobe: 5 | QPS: 5150 | Recall@10: 62.3%
                # To:   [SIVF] nprobe: 5, QPS: 5150, Recall@10: 62.3%, nlist: 1024
                CLEAN_LINE=$(echo "$line" | sed 's/|/,/g')
                echo "$(date +%H:%M:%S), $CLEAN_LINE, nlist: $nlist" >> $LOG_FILE
            fi
        done <<< "$OUTPUT"
        
    done
done

echo "--------------------------------------------------------"
echo "Grid Search Complete. Data saved to $LOG_FILE"