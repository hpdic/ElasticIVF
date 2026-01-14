// sift_loader.h
#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <sys/stat.h>

inline bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

// 读取 fvecs 格式: [dim] [v1...] [dim] [v2...]
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }

    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        exit(1);
    }

    int d;
    if (fread(&d, 1, sizeof(int), f) != sizeof(int)) {
        fprintf(stderr, "Error reading dimension\n");
        exit(1);
    }
    *d_out = (size_t)d;

    // 计算向量总数
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n_out = size / (sizeof(int) + d * sizeof(float));

    float* x = new float[*n_out * *d_out];
    
    // 循环读取
    size_t nr = 0;
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) {
            fprintf(stderr, "Error at vector %zu: dim %d != %d\n", i, d_check, d);
            exit(1);
        }
        nr += fread(x + i * d, sizeof(float), d, f);
    }
    
    fclose(f);
    return x;
}