#include "sh_malloc.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                                                      \
    {                                                                                                   \
        auto status = static_cast<cudaError_t>(error);                                                  \
        if (status != cudaSuccess) {                                                                    \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(status);                                                                          \
        }                                                                                               \
    }
#endif

__global__ void test() {
    extern __shared__ char shmem[];

    init_shmem((size_t)1024);

    void *ptr1 = sh_malloc(100);
    printf("ptr1: %p\n", ptr1);
    printf("shmem: %p\n", shmem);

    assert(ptr1 == shmem + 2 * sizeof(size_t));

    for (int i = 0; i < 9; i++) {
        void *ptr = sh_malloc(100);

        assert((char *)ptr - (char *)ptr1 == 100);

        ptr1 = ptr;
    }

    // Should fail
    void *ptr = sh_malloc(100);
}

int main() {

    std::cout << "Hello, World!" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max shared memory per block: %d bytes\n", prop.sharedMemPerBlockOptin);
    size_t maxSharedMemory = prop.sharedMemPerBlockOptin;

    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(test, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemory));

    dim3 block(1);
    dim3 grid(1);

    test<<<grid, block, maxSharedMemory>>>();
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " " << __FILE__ << ":" << __LINE__ << std::endl;
        std::exit(err);
    }
}