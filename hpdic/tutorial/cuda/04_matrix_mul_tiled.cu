#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMulTiledKernel(float* A, float* B, float* C, int N) {
    // 1. 申请黑板空间 (16x16 = 256 个位置)
    __shared__ float s_A[16][16];
    __shared__ float s_B[16][16];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int col = blockIdx.x * 16 + tx;
    int row = blockIdx.y * 16 + ty;

    float sum = 0.0f;

    // 2. 核心优化：滑动窗口 (以 16 为步长)
    for (int m = 0; m < N / 16; ++m) {
        // --- 协作搬运阶段 ---
        // 每个线程从全局显存搬一个点到黑板上
        // 这里的坐标变换比较绕，就是把大矩阵里对应的小块抠出来
        s_A[ty][tx] = A[row * N + (m * 16 + tx)];
        s_B[ty][tx] = B[(m * 16 + ty) * N + col];

        // --- 强制等待阶段 ---
        // 谁也不许抢跑！必须等 256 个人全部搬完
        __syncthreads();

        // --- 疯狂计算阶段 ---
        // 现在我们在 100 倍速的黑板上做 16 次乘加
        for (int k = 0; k < 16; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // --- 再次等待阶段 ---
        // 确保大家都算完了，才能擦黑板换下一组数
        __syncthreads();
    }

    if (row < N && col < N) {
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
    matrixMulTiledKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

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
nvcc 04_matrix_mul_tiled.cu -o matrix_mul_tiled.bin
./matrix_mul_tiled.bin
 */