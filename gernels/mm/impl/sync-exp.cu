
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <vector>

#include "../../benchmark.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

__global__ void fused_saxpy(int n, float a, float a2, float *x, float *x2,
                            float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
        // __syncthreads();
    }
    if (i < n) {
        x2[i] = a2 * x2[i] + y[i];
        __syncthreads();
    }
}

int main(void) {
    int N = 1 << 20;
    float *x, *y, *x2, *d_x, *d_y, *d_x2;
    x = (float *)malloc(N * sizeof(float));
    x2 = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_x2, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        x2[i] = 227.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2, N * sizeof(float), cudaMemcpyHostToDevice);

#define THREAD_BLOCK_SIZE 64
    // Perform SAXPY on 1M elements
    auto func = [&]() {
        saxpy<<<(N + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,
                THREAD_BLOCK_SIZE>>>(
            N, 2.0f, d_x, d_y);
    };
    double time_saxpy = benchmark::benchmark(10, 1, func, 2);
    std::cout << "Time saxpy: " << time_saxpy << " s" << std::endl;
    std::cout << "GFLOPS saxpy: " << static_cast<double>(int64_t(2) * N) / (time_saxpy * 1e9) << std::endl;
    // saxpy<<<(N + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE,
    // THREAD_BLOCK_SIZE>>>(
    //    N, 2.0f, d_y, d_x2);

    cudaMemcpy(x2, d_x2, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
