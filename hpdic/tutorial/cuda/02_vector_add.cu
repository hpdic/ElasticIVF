#include <stdio.h>
#include <cuda_runtime.h>

// GPU 核函数
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    // 【核心难点】：计算当前线程对应的数组下标
    // blockDim.x : 一个块里有多少线程
    // blockIdx.x : 我是第几个块
    // threadIdx.x: 我是块里的第几个线程
    // 全局索引 i = 块ID * 块大小 + 线程ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止越界（因为线程总数可能会略微超过数组长度）
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);

    // 1. 在 Host (CPU) 这一侧准备数据
    //    这里我为了省事用了 managed memory (统一内存)，实际工程中通常用 cudaMalloc
    //    但为了初学不被 cudaMemcpy 搞晕，我们先用这个“魔法”
    float *h_a, *h_b, *h_c;
    
    // cudaMallocManaged 分配的内存，CPU 和 GPU 都能直接访问（自动搬运）
    cudaMallocManaged(&h_a, bytes);
    cudaMallocManaged(&h_b, bytes);
    cudaMallocManaged(&h_c, bytes);

    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. 设计并行规模
    int threadsPerBlock = 256; // 每个块通常设定 256 或 512 个线程
    // 块的数量 = 总任务数 / 每个块的线程数 (向上取整)
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks and %d threads per block\n", blocksPerGrid, threadsPerBlock);

    // 3. 启动核函数
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(h_a, h_b, h_c, n);

    // 4. 等待 GPU 完成
    cudaDeviceSynchronize();

    // 5. 验证结果
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at %d: %f\n", i, h_c[i]);
            success = false;
            break;
        }
    }

    if (success) printf("Success! 1.0 + 2.0 = 3.0\n");

    // 6. 释放显存
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);

    return 0;
}

/**
nvcc 02_vector_add.cu -o vector_add.bin
./vector_add.bin
 */