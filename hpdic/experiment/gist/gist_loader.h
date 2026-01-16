/**
 * sift_loader.h
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Utility functions for loading standard ANN benchmark datasets (e.g., SIFT1M, GIST1M)
 * stored in the corpus-texmex .fvecs and .ivecs binary formats.
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <iostream>

/**
 * Check if a file exists on the filesystem.
 *
 * @param name Path to the file.
 * @return true if the file exists, false otherwise.
 */
inline bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

/**
 * Read a dataset file in .fvecs format (floating point vectors).
 *
 * Format structure:
 * [dim (int32)] [element 1 (float)] ... [element dim (float)]
 * [dim (int32)] [element 1 (float)] ... [element dim (float)]
 * ...
 *
 * @param fname Path to the .fvecs file.
 * @param d_out Output pointer for the vector dimension.
 * @param n_out Output pointer for the number of vectors.
 * @return Pointer to the allocated array containing the flattened vector data.
 */
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }
    
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "[Error] Could not open %s\n", fname);
        exit(1);
    }

    int d;
    // Read the dimension of the first vector to determine 'd'
    if (fread(&d, 1, sizeof(int), f) != sizeof(int)) {
        fprintf(stderr, "[Error] Failed to read dimension from %s\n", fname);
        exit(1);
    }
    *d_out = (size_t)d;

    // Calculate the total number of vectors based on file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Each vector occupies: sizeof(int) header + d * sizeof(float) data
    *n_out = size / (sizeof(int) + d * sizeof(float));

    // Allocate memory for the raw data
    float* x = new float[*n_out * *d_out];
    size_t nr = 0;

    // Iterate through the file to read all vectors
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        // Verify consistency: Every vector must specify the same dimension
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) {
            fprintf(stderr, "[Error] Dimension mismatch at vector %zu: expected %d, got %d\n", i, d, d_check);
            exit(1);
        }
        // Read the vector data
        nr += fread(x + i * d, sizeof(float), d, f);
    }
    
    fclose(f);
    return x;
}

/**
 * Read a dataset file in .ivecs format (integer vectors).
 * Typically used for ground truth indices in ANN benchmarks.
 *
 * Format structure:
 * [dim (int32)] [element 1 (int32)] ... [element dim (int32)]
 * ...
 *
 * @param fname Path to the .ivecs file.
 * @param d_out Output pointer for the vector dimension (usually 'k' neighbors).
 * @param n_out Output pointer for the number of vectors.
 * @return Pointer to the allocated array containing the flattened integer data.
 */
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    if (!file_exists(fname)) {
        fprintf(stderr, "[Error] File not found: %s\n", fname);
        exit(1);
    }

    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "[Error] Could not open %s\n", fname);
        exit(1);
    }

    int d;
    // Read the dimension of the first vector
    if (fread(&d, 1, sizeof(int), f) != sizeof(int)) {
        fprintf(stderr, "[Error] Failed to read dimension from %s\n", fname);
        exit(1);
    }
    *d_out = (size_t)d;

    // Calculate total number of vectors
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Each vector occupies: sizeof(int) header + d * sizeof(int) data
    *n_out = size / (sizeof(int) + d * sizeof(int));

    int* x = new int[*n_out * *d_out];

    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) {
            fprintf(stderr, "[Error] Dimension mismatch at vector %zu: expected %d, got %d\n", i, d, d_check);
            exit(1);
        }
        fread(x + i * d, sizeof(int), d, f);
    }

    fclose(f);
    return x;
}