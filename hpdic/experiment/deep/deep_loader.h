/**
 * deep_loader.h
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 * 
 * Supports .fbin and .bin formats used in Deep1B/BigANN datasets.
 * Format: [n_vectors (int32)] [dim (int32)] [data (row-major)]
 */

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

/**
 * Load .fbin file (Base and Query vectors)
 * Header: int32(n), int32(d)
 */
float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Could not open %s\n", fname); exit(1); }

    int n_in, d_in;
    // Read Header: Number of vectors (N) and Dimension (D)
    fread(&n_in, sizeof(int), 1, f);
    fread(&d_in, sizeof(int), 1, f);

    *n_out = (size_t)n_in;
    *d_out = (size_t)d_in;

    std::cout << "[Loader] Reading .fbin: N=" << *n_out << ", D=" << *d_out << std::endl;

    float* data = new float[*n_out * *d_out];
    
    // Read the contiguous data block directly
    size_t elements_read = fread(data, sizeof(float), *n_out * *d_out, f);
    
    if (elements_read != *n_out * *d_out) {
        fprintf(stderr, "[Error] Partial read: expected %zu elements, got %zu\n", *n_out * *d_out, elements_read);
    }

    fclose(f);
    return data;
}

/**
 * Load .bin file (Ground Truth indices)
 * Header: int32(n), int32(k) -> assuming standard BigANN GT format
 */
int* ibin_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Could not open %s\n", fname); exit(1); }

    int n_in, d_in;
    // Read Header
    fread(&n_in, sizeof(int), 1, f);
    fread(&d_in, sizeof(int), 1, f); // Here d_in is usually K (e.g., 100 or 1000)

    *n_out = (size_t)n_in;
    *d_out = (size_t)d_in;

    std::cout << "[Loader] Reading GT .bin: Queries=" << *n_out << ", K=" << *d_out << std::endl;

    int* data = new int[*n_out * *d_out];
    fread(data, sizeof(int), *n_out * *d_out, f);
    fclose(f);
    return data;
}