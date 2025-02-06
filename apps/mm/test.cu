
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "benchmark.h"
#include "sgemm_device.cuh"
#include "shims.cuh"

__host__ void querySharedMemory(int deviceId = 0) {
    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, deviceId);

    if (error != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(error));
        return;
    }

    // Get shared memory per block in bytes
    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;

    // Get maximum shared memory per block in bytes (for newer architectures)
    size_t sharedMemPerBlockOptin = deviceProp.sharedMemPerBlockOptin;

    printf("Device %d: %s\n", deviceId, deviceProp.name);
    printf("Default Shared Memory per Block: %zu bytes (%.2f KB)\n",
           sharedMemPerBlock, sharedMemPerBlock / 1024.0);
    printf("Maximum Shared Memory per Block: %zu bytes (%.2f KB)\n",
           sharedMemPerBlockOptin, sharedMemPerBlockOptin / 1024.0);

    // Get current shared memory configuration
    int currentSharedMem;
    cudaDeviceGetAttribute(&currentSharedMem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           deviceId);
    printf("Current Shared Memory Configuration: %zu bytes (%.2f KB)\n",
           currentSharedMem, currentSharedMem / 1024.0);
}

int main(int argc, char **argv) {
    querySharedMemory();
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int row_major = true;
    float alpha = 0.5f;
    float beta = 3.0f;

    using MatrixTypeA = impl::MatrixGPU<M, K, K, !row_major>;
    MatrixTypeA a;
    a.ascending();
    using MatrixTypeB = impl::MatrixGPU<K, N, N, row_major>;
    MatrixTypeB b;
    b.ascending();
    using MatrixTypeC = impl::MatrixGPU<M, N, N, row_major>;
    MatrixTypeC c;
    c.vvals(0.0f);

    runSgemmWarptiling(M, N, K, alpha, a.data, b.data, beta, c.data);
    auto ref_c = c.get();

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&cublasH));

    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(cublasH, stream));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    c.vvals(0.0f);
    CUBLAS_CHECK_AND_EXIT(cublasSgemm(cublasH, transa, transb, M, N, K,
                                      &alpha, a.data, K, b.data, N,
                                      &beta, c.data, N));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    auto ref_blas = c.get();

    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    c.vvals(0.0f);
    runSgemmGern(a, b, c, alpha, beta);
    auto ref_gern = c.get();

    for (int i = 0; i < M * N; i++) {
        // std::cout << i << std::endl;
        // std::cout << ref_c.data[i] << std::endl;
        // std::cout << ref_blas.data[i] << std::endl;
        assert(ref_c.data[i] - ref_blas.data[i] < 0.00001f);
        assert(ref_gern.data[i] - ref_blas.data[i] < 0.00001f);
    }
}