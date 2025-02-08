
// Directly taken from https://github.com/siboehm/SGEMM_CUDA!

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
inline __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
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

        // reinterpret_cast<float4 *>(
        //     &As[(innerRowA + offset) * BM + innerRowA * 4])[0] =
        //     reinterpret_cast<const float4 *>(
        //         &A[(innerRowA + offset) * K + innerColA * 4])[0];
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
processFromSmem(float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {

    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // populate registers for whole warptile
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                    As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                       threadRowInWarp * TM + i];
            }
        }

        // loadIntoReg(As, )

        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                       threadColInWarp * TN + i];
            }
        }

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

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(  // Immediately amenable to the GERN interface!
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                            TN>(threadResults, As, Bs, warpRow, warpCol,
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

namespace blk {

// This is coloumn major load, GMEM is row major anyway
template<
    const int NumRow,
    const int NumCol,
    const int NUM_THREADS,
    typename T>
__device__ void loadIntoSharedNew(float *As, const T &A_DS, uint x, uint y) {

    constexpr int K = A_DS.col;
    const uint innerRow = threadIdx.x / (NumCol / 4);
    const uint innerCol = threadIdx.x % (NumCol / 4);
    auto A = A_DS.data + ((x + innerRow) * K + y + innerCol * 4);  // get to the start.

    if constexpr (!A_DS.row_major) {  // load up in coloumn major
        constexpr uint rowStride = (NUM_THREADS * 4) / NumCol;
        for (uint offset = 0; offset + rowStride <= NumRow; offset += rowStride) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                A)[0];
            As[(innerCol * 4 + 0) * NumRow + innerRow + offset] = tmp.x;
            As[(innerCol * 4 + 1) * NumRow + innerRow + offset] = tmp.y;
            As[(innerCol * 4 + 2) * NumRow + innerRow + offset] = tmp.z;
            As[(innerCol * 4 + 3) * NumRow + innerRow + offset] = tmp.w;
            A += K;
        }
    } else {
        constexpr uint rowStrideB = NUM_THREADS / (NumCol / 4);
        for (uint offset = 0; offset + rowStrideB <= NumRow; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &As[(innerRow + offset) * NumCol + innerCol * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &A[(offset * K)])[0];
        }
    }
}

template<const int WMITER, const int WNITER, const int TM, const int TN>
__device__ void inner_kernel(const float *regM, const float *regN, float *result) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // calculate per-thread results
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    result[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                           (wSubColIdx * TN) + resIdxN] +=
                        regM[wSubRowIdx * TM + resIdxM] *
                        regN[wSubColIdx * TN + resIdxN];
                }
            }
        }
    }
}

template<const int BM, const int BN, const int BK, const int WM, const int WN,
         const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
         const int TM, const int TN>
__device__ void
matrix_multiply(float *threadResults, const float *As,
                const float *Bs) {

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;          // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);  // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);  // i/4

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE;  // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

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
        inner_kernel<WMITER, WNITER, TM, TN>(regM, regN, threadResults);
    }
}

// template<int64_t ti, int64_t tj, int64_t tk,
//          typename T1,
//          typename T2,
//          typename T3>
// __global__ void gern_gen(T1 A, T2 B, T3 C) {

//     int64_t _gern_i_2_7_13_19 = ((blockIdx.y * ti) + 0);
//     int64_t _gern_j_3_8_14 = ((blockIdx.x * tj) + 0);
//     int64_t _gern_j_3_8 = _gern_j_3_8_14;
//     constexpr int64_t _gern_tj_5_11 = tj;
//     int64_t _gern_i_2_7 = _gern_i_2_7_13_19;
//     constexpr int64_t _gern_ti_4_10 = ti;

//     float threadResults[ti * tj] = {0.0};

//     for (int64_t _gern_k_1_9 = 0; (_gern_k_1_9 < C.reduce); _gern_k_1_9 = (_gern_k_1_9 + tk)) {

//         int64_t _gern_j_3 = _gern_j_3_8_14;
//         constexpr int64_t _gern_tj_5 = tj;
//         int64_t _gern_i_2 = _gern_i_2_7_13_19;
//         constexpr int64_t _gern_ti_4 = ti;

//         constexpr int64_t _gern_k_1 = _gern_k_1_9;
//         constexpr int64_t _gern_tk_6 = tk;

//         auto _query_A_27 = A.template query_new<_gern_ti_4, _gern_tk_6>(_gern_i_2, _gern_k_1);

//         auto _query_B_28 = B.template query_new<_gern_tk_6, _gern_tj_5>(_gern_k_1, _gern_j_3);

//         matrix_multiply<_gern_k_1>(_query_A_27, _query_B_28, _query_C_26);
//     }

//     C.template insert_new(_gern_i_2_7, _gern_j_3_8, _query_C_26);
// }

template<const int BM, const int BN,
         const int BK, const int WM,
         const int WN,
         const int WNITER,
         const int TM, const int TN,
         int K,
         typename T>
__device__ void
matrix_multiply_shared(T C_DS, const float *As,
                       const float *Bs) {

    constexpr int64_t M = C_DS.row;
    constexpr int64_t N = C_DS.col;

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

    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;
}

}  // namespace blk

template<typename T1,
         typename T2,
         typename T3, const int BM, const int BN, const int BK, const int WM, const int WN,
         const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void sgemmGernShared(T1 A_DS, T2 B_DS, T3 C_DS, float alpha, float beta) {

    constexpr int64_t M = A_DS.row;
    constexpr int64_t K = A_DS.col;
    constexpr int64_t N = B_DS.col;

    float *C = C_DS.data;

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

    // allocate space for the current blocktile in SMEM

    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;
    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level

    // outer-most loop over block tiles
    float *C_temp = cRow * BM * N + cCol * BN;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];
        blk::loadIntoSharedNew<BM, BK, NUM_THREADS>(As, A_DS, cRow * BM, bkIdx);
        blk::loadIntoSharedNew<BK, BN, NUM_THREADS>(Bs, B_DS, bkIdx, cCol * BN);
        __syncthreads();
        blk::matrix_multiply<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                             TN>(threadResults, As, Bs);
        // blk::matrix_multiply_shared<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
        //                             TN>(C_temp, As, Bs);

        __syncthreads();
    }

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;          // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);  // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);  // i/4

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