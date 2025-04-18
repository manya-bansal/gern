#include "../../benchmark.h"
#include "column-major.h"
#include "matrix-gpu.h"
#include "matrix_multiply.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
const int WARPSIZE = 32;  // warpSize is not constexpr

namespace wt {
template<const int BM, const int BN, const int BK, const int rowStrideA,
         const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const float4 tmp = reinterpret_cast<const float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        // float4 tmp;
        // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
        // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
        //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
}

template<const int BM, const int BN, const int BK, const int WM, const int WN,
         const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
         const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // populate registers for whole warptile
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                    As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                       threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                       threadColInWarp * TN + i];
            }
        }

        // execute warptile matmul
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // calculate per-thread results
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      (wSubColIdx * TN) + resIdxN] +=
                            regM[wSubRowIdx * TM + resIdxM] *
                            regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
    }
}

}  // namespace wt

template<const int BM, const int BN, const int BK, const int WM, const int WN,
         const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE;  // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;  // 64/2=32
    constexpr uint WSUBN = WN / WNITER;  // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;          // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);  // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);  // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                            TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                                threadRowInWarp, threadColInWarp);
        A += BK;      // move BK columns to right
        B += BK * N;  // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    // write back
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template<const int BM, const int BN, const int BK, const int WM, const int WN,
         const int WNITER, const int TM, const int TN, const int NUM_THREADS,
         typename CT, typename AT, typename BT>
__global__ void __launch_bounds__(NUM_THREADS)
    gernel_mine_sgemmWarptiling(int M, int N, int K, float alpha, AT A_ds, BT B_ds,
                                float beta, CT C_ds) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    float *C = C_ds.data;
    float *A = A_ds.data;
    float *B = B_ds.data;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE;  // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;  // 64/2=32
    constexpr uint WSUBN = WN / WNITER;  // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;          // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);  // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);  // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column

    auto a_q = A_ds.template query_global_2_global<BM, BK>(cRow * BM, 0);
    A = a_q.data;
    auto b_q = B_ds.template query_global_2_global<BK, BN>(0, cCol * BN);
    B = b_q.data;

    auto c_q = C_ds.template query_global_2_global<BM, BN>(cRow * BM, cCol * BN);
    auto c_w_q = c_q.template query_global_2_global<WM, WN>(warpRow * WM, warpCol * WN);
    C = c_w_q.data;

    // Move C_ptr to warp's output tile
    // C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    // float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level
    // float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    auto c_reg = c_q.template query_2_reg_no_vector_zero<WMITER * TM, WNITER * TN>(0, 0);
    float *threadResults = c_reg.array;

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        auto a_s_mine = a_q.template query_global_2_shared_vec_t<BM, BK, NUM_THREADS>(0, bkIdx, As);

        // Why is my function slower??
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
            // &b_q(innerRowB + offset + bkIdx, innerColB * 4))[0];
        }

        impl::MatrixGPU<BK, BN, BN, 1> b_s_mine(Bs);

        // auto b_s_mine = b_q.template query_global_2_shared_vec<BK, BN, NUM_THREADS>(bkIdx, 0, Bs);

        __syncthreads();

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // Transposed A
            auto a_s_warp = a_s_mine.template query_global_2_global<BK, WM>(dotIdx, 0);
            auto b_s_warp = b_s_mine.template query_global_2_global<BK, WN>(dotIdx, warpCol * WN + threadColInWarp * TN);

            StaticMatrixNoVector<WMITER, TM> a_reg;

            StaticMatrixNoVector<WNITER, TN> b_reg;

            // This weird structure just has to do with column major??
            // If I just model that i SHOULD BE OKAY
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i = 0; i < TM; ++i) {
                    a_reg(wSubRowIdx, i) =
                        a_s_warp(0, warpRow * WM + wSubRowIdx * WSUBM +
                                        threadRowInWarp * TM + i);
                }
            }

            // Manya: this just staging b early.

            // Manya: this pulling one line of B in.
            // WNITER: HOW MANY OF B DO WE HAVE?
            // WHAT'S THE LENGTH OF EACH?
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                auto b_s_warp_tile = b_s_warp.template query_global_2_global<BK, WN>(0, wSubColIdx * WSUBN);
                for (uint i = 0; i < TN; ++i) {
                    b_reg(wSubColIdx, i) =
                        b_s_warp_tile(0, i);
                }
            }

            // ABOVE IS EQUIVALENT TO:
            // It's just producing a tile output
            // So, truly, breg is ONE COLOUMN of size [1, TN * WNITER]
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                    b_reg.array[i + wSubColIdx * TN] = b_s_warp(0, (wSubColIdx * WSUBN + i));
                }
            }

            // execute warptile matmul
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {

                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {

                    // then there is a threads in a warp tile.....
                    auto c_thread = c_reg.template get_view<TM, TN>(wSubRowIdx * TM, wSubColIdx * TN);
                    auto a_thread_reg = a_reg.template get_view<1, TM>(wSubRowIdx, 0);
                    auto b_thread_reg = b_reg.template get_view<1, TN>(wSubColIdx, 0);

                    matrix_multiply_reg_flat_T<1>(a_thread_reg, b_thread_reg, c_thread);

                    // // calculate per-thread results
                    // for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    //     for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    //         // assert(&a_thread_reg(0, resIdxM) == &a_reg(wSubRowIdx, resIdxM));
                    //         // assert(&b_thread_reg(0, resIdxN) == &b_reg(wSubColIdx, resIdxN));
                    //         c_thread(resIdxM, resIdxN) +=
                    //             a_thread_reg(0, resIdxM) *
                    //             b_thread_reg(0, resIdxN);
                    //     }
                    // }
                }
            }
        }

        // A += BK;      // move BK columns to right
        B += BK * N;  // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            // float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            // float *C_interim = &c_w_q(wSubRowIdx * WSUBM, wSubColIdx * WSUBN);
            auto warp_query_c = c_w_q.template query_global_2_global<WM, WN>(wSubRowIdx * WSUBM, wSubColIdx * WSUBN);
            float *C_interim = warp_query_c.data;

            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    float4 tmp = reinterpret_cast<float4 *>(
                        &warp_query_c.data[(threadRowInWarp * TM + resIdxM) * N +
                                           threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    // write back
                    reinterpret_cast<float4 *>(
                        &warp_query_c.data[(threadRowInWarp * TM + resIdxM) * N +
                                           threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

int main() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    // Settings for A100
    // const uint K10_NUM_THREADS = 128;
    // const uint K10_BN = 128;
    // const uint K10_BM = 64;
    // const uint K10_BK = 16;
    // const uint K10_WN = 64;
    // const uint K10_WM = 32;
    // const uint K10_WNITER = 1;
    // const uint K10_TN = 4;
    // const uint K10_TM = 4;
    // Settings for A6000
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

    // impl::SharedMemoryManager smem_manager;
    int32_t offset = 0;
    impl::MatrixGPU<M, K, K, 1> A;
    offset += K10_BM * K * sizeof(float);
    // impl::MatrixGPU<K, N, N, 1> B;
    impl::ColumnMajorMatrix<K, N> B(offset);

    offset += 32 * 32 * sizeof(float);
    impl::MatrixGPU<M, N, N, 1> C(offset);

    A.ascending();
    B.ascending();
    C.vvals(0.0f);

    float alpha = 1.0f;
    float beta = 0.0f;

    dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));

    impl::MatrixGPU<M, N, N, 1> C_naive(offset);
    C_naive.vvals(0.0f);

    auto func_original = [&]() {
        sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                        K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A.data, B.data, beta, C_naive.data);
    };

    double time_original = benchmark::benchmark(10, 1, func_original, 2);
    std::cout << "Time original: " << time_original << " ms" << std::endl;
    std::cout << "GFLOPS original: " << static_cast<double>(int64_t(2) * M * N * K) / (time_original * 1e9) << std::endl;
    auto C_complex_original = C_naive.get();

    auto func = [&]() {
        gernel_mine_sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                                    K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    };

    double time = benchmark::benchmark(10, 1, func, 2);
    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "GFLOPS: " << static_cast<double>(int64_t(2) * M * N * K) / (time * 1e9) << std::endl;
    auto C_complex = C.get();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            assert(C_complex(i, j) == C_complex_original(i, j));
        }
    }
}

// benchmark(func);
