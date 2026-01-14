#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <iostream>

inline bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) { fprintf(stderr, "File not found: %s\n", fname); exit(1); }
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Open failed: %s\n", fname); exit(1); }
    int d;
    fread(&d, 1, sizeof(int), f);
    *d_out = (size_t)d;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n_out = size / (sizeof(int) + d * sizeof(float));
    float* x = new float[*n_out * *d_out];
    size_t nr = 0;
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) exit(1);
        nr += fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) { fprintf(stderr, "File not found: %s\n", fname); exit(1); }
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Open failed: %s\n", fname); exit(1); }
    int d;
    fread(&d, 1, sizeof(int), f);
    *d_out = (size_t)d;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n_out = size / (sizeof(int) + d * sizeof(int));
    int* x = new int[*n_out * *d_out];
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) exit(1);
        fread(x + i * d, sizeof(int), d, f);
    }
    fclose(f);
    return x;
}