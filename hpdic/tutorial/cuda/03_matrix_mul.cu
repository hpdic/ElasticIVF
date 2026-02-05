#include <stdio.h>
#include <cuda_runtime.h>

// 核心 Kernel：每个线程计算 C[row][col] 的值
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    // 1. 计算 2D 坐标
    // blockIdx.x 和 threadIdx.x 对应列 (col)
    // blockIdx.y 和 threadIdx.y 对应行 (row)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. 边界检查 (防止溢出)
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // 3. 执行点积 (Row of A * Col of B)
        // 这是一个普通的 CPU 循环，但是由 GPU 线程并行执行
        for (int k = 0; k < N; ++k) {
            // A 是按行存的，B 是按行存的
            // A[row][k] * B[k][col]
            sum += A[row * N + k] * B[k * N + col];
        }

        // 4. 写回结果
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024; // 矩阵大小 1024 x 1024
    size_t bytes = N * N * sizeof(float);

    // Host 内存分配
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);

    // 初始化数据 (全 1.0)
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 拷贝数据 H -> D
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // --- 关键设置：2D Grid 和 2D Block ---
    // 这里的 (16, 16) 就是你刚才算的一个 Block 256 线程
    dim3 threadsPerBlock(16, 16); 
    
    // Grid 也要覆盖整个 2D 平面
    // (1024 + 15) / 16 = 64
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Kernel Launch Configuration:\n");
    printf("Grid Size : {%d, %d, 1}\n", numBlocks.x, numBlocks.y);
    printf("Block Size: {%d, %d, 1}\n", threadsPerBlock.x, threadsPerBlock.y);

    // 启动核函数
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 拷贝回结果 D -> H
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 验证结果 (C[0][0] 应该是 1024 * 1.0 * 1.0 = 1024)
    printf("Result check: C[0] = %f (Expected: 1024.00)\n", h_C[0]);

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}

/**
nvcc 03_matrix_mul.cu -o matrix_mul.bin
./matrix_mul.bin
 */