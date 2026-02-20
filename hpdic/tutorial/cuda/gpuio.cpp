#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <string>
#include <cstdlib>

void run_storage_benchmark() {
    size_t size = 1073741824; 
    
    void* d_buf;
    cudaMalloc(&d_buf, size);
    
    void* h_buf;
    cudaHostAlloc(&h_buf, size, cudaHostAllocDefault);
    
    const char* home_dir = getenv("HOME");
    std::string file_path_str = std::string(home_dir) + "/hpdic/data.bin";
    const char* file_path = file_path_str.c_str();
    
    auto start_baseline = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost);
    int fd_base = open(file_path, O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
    write(fd_base, h_buf, size);
    close(fd_base);
    
    auto end_baseline = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_baseline = end_baseline - start_baseline;
    double bw_baseline = (size / 1024.0 / 1024.0 / 1024.0) / diff_baseline.count();
    
    std::cout << "Baseline CPU Buffer Time: " << diff_baseline.count() << " s\n";
    std::cout << "Baseline Bandwidth: " << bw_baseline << " GB/s\n\n";
    
    cuFileDriverOpen();
    int fd_gds = open(file_path, O_CREAT | O_WRONLY | O_DIRECT | O_TRUNC, 0664);
    
    CUfileDescr_t cf_descr;
    cf_descr.handle.fd = fd_gds;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    CUfileHandle_t cf_handle;
    cuFileHandleRegister(&cf_handle, &cf_descr);
    
    auto start_gds = std::chrono::high_resolution_clock::now();
    
    cuFileWrite(cf_handle, d_buf, size, 0, 0);
    
    auto end_gds = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gds = end_gds - start_gds;
    double bw_gds = (size / 1024.0 / 1024.0 / 1024.0) / diff_gds.count();
    
    std::cout << "GPUDirect Storage Time: " << diff_gds.count() << " s\n";
    std::cout << "GPUDirect Storage Bandwidth: " << bw_gds << " GB/s\n";
    
    cuFileHandleDeregister(cf_handle);
    close(fd_gds);
    cuFileDriverClose();
    
    cudaFree(d_buf);
    cudaFreeHost(h_buf);
}

int main() {
    run_storage_benchmark();
    return 0;
}

/**
nvcc gpuio.cpp -o gpuio.bin -lcudart -lcufile
./gpuio.bin
 */