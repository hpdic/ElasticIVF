/**
 * sift_loader.h
 *
 * Author: Dongfang Zhao
 * Email:  dzhao@uw.edu
 *
 * Utility functions for loading standard ANN benchmark datasets (SIFT, GIST)
 * stored in .fvecs and .ivecs binary formats.
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <sys/stat.h>

/**
 * Check if a file exists on the filesystem.
 */
inline bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

/**
 * Read a file in .fvecs format.
 * Format structure: [dim (int)] [vector_data (float * dim)] ... repeated n times
 *
 * @param fname Path to the .fvecs file
 * @param d_out Pointer to store the dimension of vectors
 * @param n_out Pointer to store the number of vectors
 * @return Pointer to the allocated float array containing raw vector data
 */
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

    // Calculate total number of vectors based on file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n_out = size / (sizeof(int) + d * sizeof(float));

    float* x = new float[*n_out * *d_out];
    
    // Read vectors sequentially
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

/**
 * Read a file in .ivecs format (typically used for ground truth indices).
 * Format structure: [dim (int)] [index_data (int * dim)] ... repeated n times
 *
 * @param fname Path to the .ivecs file
 * @param d_out Pointer to store the dimension (usually k for ground truth)
 * @param n_out Pointer to store the number of queries
 * @return Pointer to the allocated int array containing raw index data
 */
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
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

    // Calculate total number of vectors
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    // Note: The element size here is sizeof(int)
    *n_out = size / (sizeof(int) + d * sizeof(int)); 

    int* x = new int[*n_out * *d_out];
    
    size_t nr = 0;
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        if (d_check != d) {
            fprintf(stderr, "Error at vector %zu: dim %d != %d\n", i, d_check, d);
            exit(1);
        }
        nr += fread(x + i * d, sizeof(int), d, f);
    }
    
    fclose(f);
    return x;
}