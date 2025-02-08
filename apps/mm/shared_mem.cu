#include <cuda_runtime.h>
#include <iostream>

// Kernel that attempts to use an extremely large shared memory allocation
__global__ void excessSharedMemKernel(float *d_out, int size) {
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        shared_mem[tid % blockDim.x] = 1.0f;  // Simple write
        d_out[tid] = shared_mem[tid % blockDim.x];
    }
}

int main() {
    int N = 1024;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_out = new float[N];

    // Allocate device memory
    float *d_out;
    cudaMalloc(&d_out, bytes);

    // Check GPU shared memory limits
    int sharedMemPerBlockDefault, sharedMemMaxPossible;
    cudaDeviceGetAttribute(&sharedMemPerBlockDefault, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaDeviceGetAttribute(&sharedMemMaxPossible, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);

    std::cout << "Default max shared memory per block: " << sharedMemPerBlockDefault << " bytes\n";
    std::cout << "Absolute max shared memory per multiprocessor: " << sharedMemMaxPossible << " bytes\n";

    int threadsPerBlock = 256;
    int excessiveSharedMemSize = sharedMemPerBlockDefault + 1;  // Try exceeding the default limit

    // Attempt to set a higher shared memory limit
    cudaError_t err = cudaFuncSetAttribute(excessSharedMemKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, excessiveSharedMemSize);
    if (err != cudaSuccess) {
        std::cout << "Failed to increase shared memory limit: " << cudaGetErrorString(err) << "\n";
    }

    // Launch kernel
    excessSharedMemKernel<<<1, threadsPerBlock, excessiveSharedMemSize>>>(d_out, N);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error after kernel launch: " << cudaGetErrorString(err) << "\n";
    }

    // Cleanup
    cudaFree(d_out);
    delete[] h_out;

    return 0;
}
