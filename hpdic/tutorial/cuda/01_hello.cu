#include <stdio.h>
#include <cuda_runtime.h>

// 1. __global__ 告诉编译器：
//    这个函数在 GPU (Device) 上跑，但是由 CPU (Host) 负责调用
__global__ void hello_kernel() {
    // threadIdx 是 CUDA 内置变量，代表当前线程在块内的 ID
    // blockIdx 是 当前块 ID
    printf("GPU: Hello from Block %d, Thread %d!\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("CPU: Preparing to launch kernel...\n");

    // 2. 核函数调用语法 <<<GridDim, BlockDim>>>
    //    这里意思是：启动 2 个 Block，每个 Block 里有 4 个 Thread
    //    总共启动 2 * 4 = 8 个 GPU 线程
    hello_kernel<<<2, 4>>>();

    // 3. 【关键点】 CPU 和 GPU 是异步的！
    //    CPU 发出发射指令后，不会等 GPU 跑完，而是直接往下走。
    //    如果没有下面这行，CPU 可能直接 return 0 退出了，
    //    导致 GPU 的 printf 还没来得及输出，程序就被杀掉了。
    cudaDeviceSynchronize(); 

    printf("CPU: All done!\n");
    return 0;
}

/**
nvcc 01_hello.cu -o hello.bin
./hello.bin
 */