#include "impl/gpu-matrix-const.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "benchmark.h"
#include "sgemm_device.cuh"
#include "shims.h"

constexpr int warm_up_runs = 10;
constexpr int kernel_repeats = 10;

int main(int argc, char **argv) {
    long max_size = 5120;

    if (argc < 2) {
        std::cout << "Need to specify an implementation! [cublas, device]" << std::endl;
        return 1;
    }

    std::string impl = argv[1];

    if (argc > 2) {
        max_size = std::stoi(argv[2]);
    }

    float alpha = 0.5, beta = 3.0;  // GEMM input parameters, C=α*AB+β*C

    float *A = nullptr, *B = nullptr, *C = nullptr,
          *C_ref = nullptr;  // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr,
          *dC_ref = nullptr;  // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    input_gen::fillArrayWithRandomNumbers(A, max_size * max_size);
    input_gen::fillArrayWithRandomNumbers(B, max_size * max_size);
    input_gen::fillArrayWithRandomNumbers(C, max_size * max_size);

    CUDA_CHECK_AND_EXIT(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    CUDA_CHECK_AND_EXIT(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    CUDA_CHECK_AND_EXIT(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    CUDA_CHECK_AND_EXIT(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    CUDA_CHECK_AND_EXIT(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                                   cudaMemcpyHostToDevice));

    if (impl == "cublas") {
        cublasHandle_t cublasH = NULL;
        cudaStream_t stream = NULL;

        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;

        CUBLAS_CHECK_AND_EXIT(cublasCreate(&cublasH));
        CUBLAS_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK_AND_EXIT(cublasSetStream(cublasH, stream));

        double time = benchmark::measure::execution(
            [&](cudaStream_t stream) {
                // kernel<<<grid, block_mm_type::block_dim, block_mm_type::shared_memory_size, stream>>>(
                //     tensor_a, tensor_b, tensor_c, alpha, beta);
                cublasSgemm(cublasH, transa, transb, max_size, max_size, max_size, &alpha, dA, max_size, dB, max_size,
                            &beta, dC, max_size);
            },
            warm_up_runs,
            kernel_repeats,
            stream);

        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        double gflops = (2.0 * max_size * max_size * max_size) * 1e-9;
        std::cout << gflops / (time / 1000) << std::endl;

    } else if (impl == "device") {
        runSgemmWarptiling(max_size, max_size, max_size, alpha, dA, dB, beta, dC, true);
    } else if (impl == "gern") {
        constexpr int M = 16384;
        constexpr int N = 16384;
        constexpr int K = 16384;
        constexpr int dummy = 2;
        float alpha = 0.5f;
        float beta = 3.0f;

        using MatrixTypeA = impl::MatrixGPU<M, K, K, dummy>;
        MatrixTypeA a;
        a.ascending();
        using MatrixTypeB = impl::MatrixGPU<K, N, N, dummy>;
        MatrixTypeB b;
        b.ascending();
        using MatrixTypeC = impl::MatrixGPU<M, N, N, dummy>;
        MatrixTypeC c;
        c.vvals(0.0f);

        runSgemmGern(a, b, c, alpha, beta, true);
    } else {
        std::cout << "Invalid Execution param [cublas, device]!" << std::endl;
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}