#!/bin/bash

# =============================================================================
# Script Name: search_t2i.sh
# Description: Automated grid search script for SIVF on the T2I-1B dataset.
#              It iterates through specified nlist and nprobe parameters
#              to benchmark QPS and Recall performance against Baselines.
# =============================================================================

# ================= Configuration =================
EXEC="$HOME/ElasticIVF/build/test_sivf_t2i_search"
LOG_FILE="$HOME/ElasticIVF/hpdic/result/grid_search_t2i"

# 1. Training Data Size (1M is standard for T2I-1M subset)
NT_TRAIN=1000000

# 2. Temporary GPU Memory (1GB is sufficient for T2I execution)
TEMP_MEM=1024

# ================= Grid Search Parameters =================

# Variable A: nlist (Number of cluster centroids)
# For T2I (200D), 4096 is usually a sweet spot for 1M vectors.
# - 1024: Might be too coarse, leading to long lists.
# - 4096: Balanced for SIVF's slab structure.
# - 8192: High precision, but might hurt QPS on lower dims compared to GIST.
NLISTS=(4096)

# Variable B: nprobe (Number of clusters to probe)
# We sweep from small (speed) to large (recall) to build the Pareto frontier.
PROBES=(16 32 64 128)

# ================= Execution =================

# Initialize Log
if [ ! -f $LOG_FILE ]; then
    echo "Timestamp, Method, nlist, nprobe, QPS, Recall" > $LOG_FILE
fi

echo "Starting Grid Search (T2I)..."
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
                # From: [SIVF] nprobe: 20 | QPS: 12345 | Recall@10: 85.123%
                # To:   [SIVF] nprobe: 20, QPS: 12345, Recall@10: 85.123%, nlist: 4096
                CLEAN_LINE=$(echo "$line" | sed 's/|/,/g')
                echo "$(date +%H:%M:%S), $CLEAN_LINE, nlist: $nlist" >> $LOG_FILE
            fi
        done <<< "$OUTPUT"
        
    done
done

echo "--------------------------------------------------------"
echo "Grid Search Complete. Data saved to $LOG_FILE"