#!/bin/bash

# 关于最大值：55000还好；60000就OOM了；
NLIST_VALS=(30000)

# nb: 包含快速值(100000), 中间值(200000), 极限值(360000)
NB_VALS=(200000)

# probes: 固定4个值，保证输出4行
PROBES="4,8,16,32,64,128"

# 3. 开始循环扫描
for nb in "${NB_VALS[@]}"; do
    for nlist in "${NLIST_VALS[@]}"; do
        echo "-----------------------------------------------------------------------"
        echo ">>> STARTING TEST: nlist=$nlist | nb=$nb | Total Vectors=$((nb * 10))"
        echo "-----------------------------------------------------------------------"
        
        # 运行 MPI 程序
        # 使用 grep 过滤日志，只显示包含表格线 "|" 且包含 "SIVF" 或 "Vanilla" 的行
        mpirun --allow-run-as-root \
            -np 2 \
            --host gpu3:2 \
            -x LD_LIBRARY_PATH \
            ~/hpdic/ElasticIVF/build/test_sivf_dino_search \
            --nlist $nlist \
            --nb $nb \
            --probes $PROBES \
            2>&1 | grep -E "\| \*\*SIVF\*\*|\| Vanilla"
        
        # 检查上一条命令的退出状态 (捕获 OOM 崩溃)
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "!!! CRASHED (Likely OOM) !!!"
        fi
        
        echo "" # 空行隔开
    done
done

# Example output:
# # -----------------------------------------------------------------------
# cc@gpu0:~/hpdic/ElasticIVF/hpdic/script$ ./search_dino.sh 
# -----------------------------------------------------------------------
# >>> STARTING TEST: nlist=30000 | nb=200000 | Total Vectors=2000000
# -----------------------------------------------------------------------
# | **SIVF** | 4      | 11.92        | 83908        | 0.8075       |
# | **SIVF** | 8      | 12.92        | 77410        | 0.8837       |
# | **SIVF** | 16     | 16.77        | 59630        | 0.9344       |
# | **SIVF** | 32     | 24.20        | 41315        | 0.9674       |
# | **SIVF** | 64     | 38.80        | 25771        | 0.9871       |
# | **SIVF** | 128    | 67.96        | 14714        | 0.9952       |
# | Vanilla  | 4      | 11.48        | 87119        | 0.8075       |
# | Vanilla  | 8      | 12.29        | 81393        | 0.8837       |
# | Vanilla  | 16     | 15.80        | 63285        | 0.9342       |
# | Vanilla  | 32     | 22.98        | 43509        | 0.9674       |
# | Vanilla  | 64     | 37.18        | 26898        | 0.9871       |
# | Vanilla  | 128    | 65.68        | 15225        | 0.9952       |