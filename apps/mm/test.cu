
#include "impl/gpu-matrix-const.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "benchmark.h"
#include "sgemm_device.cuh"

template<typename T1,
         typename T2,
         typename T3>
void runSgemmGern(T1 A, T2 B, T3 C, float alpha, float beta) {
    constexpr int64_t M = A.row;
    constexpr int64_t K = A.col;
    constexpr int64_t N = B.col;

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

    sgemmGernShared<T1, T2, T3, K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                    K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim>>>(A, B, C, alpha, beta);
}

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

    sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                    K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

int main(int argc, char **argv) {

    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
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