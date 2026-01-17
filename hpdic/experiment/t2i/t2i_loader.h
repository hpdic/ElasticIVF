/**
 * t2i_loader.h
 * 
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Utility functions for loading T2I (Text-to-Image) benchmark datasets
 * stored in the simple binary format (.fbin / .bin).
 * * Format:
 * [n_vectors (int32)] [dim (int32)] [data.......]
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <iostream>

inline bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

/**
 * Read a dataset file in .fbin format (Raw Float Binary with Header)
 * Header: [N (int32)] [D (int32)]
 */
float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }
    
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "[Error] Could not open %s\n", fname);
        exit(1);
    }

    int n_in, d_in;
    // Read Header
    if (fread(&n_in, sizeof(int), 1, f) != 1 || fread(&d_in, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "[Error] Failed to read header from %s\n", fname);
        exit(1);
    }

    *n_out = (size_t)n_in;
    *d_out = (size_t)d_in;

    std::cout << "[Loader] Header info -> N: " << *n_out << ", D: " << *d_out << std::endl;

    // Allocate memory
    // Note: Use long long for size calculation to prevent overflow
    size_t total_elements = *n_out * *d_out;
    float* data = new float[total_elements];

    // Bulk read
    size_t read_count = fread(data, sizeof(float), total_elements, f);
    if (read_count != total_elements) {
        fprintf(stderr, "[Error] Short read: expected %zu floats, got %zu\n", total_elements, read_count);
        delete[] data;
        exit(1);
    }
    
    fclose(f);
    return data;
}

/**
 * Read a ground truth file in .bin format (Raw Int Binary with Header)
 * Header: [N (int32)] [D (int32)] (D is usually K neighbors)
 */
int* ibin_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }

    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "[Error] Could not open %s\n", fname);
        exit(1);
    }

    int n_in, d_in;
    fread(&n_in, sizeof(int), 1, f);
    fread(&d_in, sizeof(int), 1, f);

    *n_out = (size_t)n_in;
    *d_out = (size_t)d_in;

    size_t total_elements = *n_out * *d_out;
    int* data = new int[total_elements];

    size_t read_count = fread(data, sizeof(int), total_elements, f);
    if (read_count != total_elements) {
        fprintf(stderr, "[Error] Short read: expected %zu ints, got %zu\n", total_elements, read_count);
        delete[] data;
        exit(1);
    }

    fclose(f);
    return data;
}