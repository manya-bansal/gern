#include "benchmark.h"
#include "impl/gpu-matrix-const.h"
#include "sgemm_device.cuh"

template<typename T1,
         typename T2,
         typename T3>
void runSgemmGern(T1 A, T2 B, T3 C, float alpha, float beta, bool benchmark = false) {
    constexpr int warm_up_runs = 10;
    constexpr int kernel_repeats = 10;

    constexpr int64_t M = A.row;
    constexpr int64_t K = A.col;
    constexpr int64_t N = B.col;

    // const uint K10_NUM_THREADS = 128;
    // const uint K10_BN = 64;
    // const uint K10_BM = 64;
    // const uint K10_BK = 16;
    // const uint K10_WN = 32;
    // const uint K10_WM = 32;
    // const uint K10_WNITER = 2;
    // const uint K10_TN = 4;
    // const uint K10_TM = 4;

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

    auto specialized = sgemmGernShared<T1, T2, T3, K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                                       K10_TN, K10_NUM_THREADS>;

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(specialized,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             cudaDevAttrMaxSharedMemoryPerBlockOptin));

    // cudaFuncAttributes attr;
    // cudaFuncGetAttributes(&attr, specialized);
    // printf("Registers per thread: %d\n", attr.numRegs);
    // printf("Shared memory per block: %d bytes\n", attr.sharedSizeBytes);
    // printf("gridDim=(%d,%d,%d), blockDim=(%d,%d,%d)\n",
    //        gridDim.x, gridDim.y, gridDim.z,
    //        blockDim.x, blockDim.y, blockDim.z);

    specialized<<<gridDim, blockDim>>>(A, B, C, alpha, beta);

    if (benchmark) {
        double time = benchmark::measure::execution(
            [&](cudaStream_t stream) {
                specialized<<<gridDim, blockDim>>>(A, B, C, alpha, beta);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Benchmark kernel launch error: %s\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
            },
            warm_up_runs,
            kernel_repeats,
            0);
        double gflops = (2.0 * M * N * K) * 1e-9;
        std::cout << gflops / (time / 1000) << std::endl;
    }
}

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C, bool benchmark = false) {
    constexpr int warm_up_runs = 10;
    constexpr int kernel_repeats = 10;

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

    if (benchmark) {
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
}
