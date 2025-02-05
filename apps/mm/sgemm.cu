#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "benchmark.h"
#include "sgemm_device.cuh"

constexpr int warm_up_runs = 10;
constexpr int kernel_repeats = 10;

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
    const uint K10_NUM_THREADS = 128;
    const uint K10_BN = 128;
    const uint K10_BM = 128;
    const uint K10_BK = 16;
    const uint K10_WN = 64;
    const uint K10_WM = 64;
    const uint K10_WNITER = 4;
    const uint K10_TN = 4;
    const uint K10_TM = 8;

    dim3 blockDim(K10_NUM_THREADS);

    constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
    static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

    // threads in warpsubtile
    static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                  0);
    constexpr uint K10_WMITER =
        (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
    // warpsubtile in warptile
    static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

    static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                  "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                  "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(K10_BN % (16 * K10_TN) == 0,
                  "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(K10_BM % (16 * K10_TM) == 0,
                  "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                  "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                  "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));

    double time = benchmark::measure::execution(
        [&](cudaStream_t stream) {
            sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                            K10_TN, K10_NUM_THREADS>
                <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        },
        warm_up_runs,
        kernel_repeats,
        0);

    double gflops = (2.0 * M * N * K) * 1e-9;
    std::cout << gflops / (time / 1000) << std::endl;
}

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
        runSgemmWarptiling(max_size, max_size, max_size, alpha, dA, dB, beta, dC);
    } else {
        std::cout << "Invalid Execution param [cublas, device]!" << std::endl;
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}