#pragma once

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

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        // transpose A while loading it
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        // advance blocktile
        A += BK;      // move BK columns to right
        B += BK * N;  // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            // perform GEMM update in reg
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
            // write back
            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
                tmp;
        }
    }
}

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template<const int BM, const int BN, const int BK, const int TM, const int TN, typename AT, typename BT, typename CT>
__global__ void gernel_mine(int M, int N, int K, float alpha, AT A_ds, BT B_ds, CT C_ds) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    float *A = A_ds.data;
    float *B = B_ds.data;
    float *C = C_ds.data;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    auto a_q = A_ds.template query_global_2_global<BM, BK>(cRow * BM, 0);
    A = a_q.data;
    auto b_q = B_ds.template query_global_2_global<BK, BN>(0, cCol * BN);
    B = b_q.data;
    auto c_q = C_ds.template query_global_2_global<BM, BN>(cRow * BM, cCol * BN);
    C = c_q.data;

    auto c_reg = c_q.template query_2_reg_no_vector_zero<TM, TN>(threadRow * TM, threadCol * TN);
    float *threadResults = c_reg.array;

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        auto a_s_mine = a_q.template query_global_2_shared_vec_t<BM, BK, (BM / TM) * (BN / TN)>(0, bkIdx, As);

        auto b_s_mine = b_q.template query_global_2_shared_vec<BK, BN, (BM / TM) * (BN / TN)>(bkIdx, 0, Bs);
        __syncthreads();

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

            auto a_reg = a_s_mine.template query_2_reg_no_vector<1, TM>(dotIdx, threadRow * TM);

            auto b_reg = b_s_mine.template query_2_reg_no_vector<1, TN>(dotIdx, threadCol * TN);

            matrix_multiply_reg_flat_T<1>(a_reg, b_reg, c_reg);
        }
        __syncthreads();
    }

    c_q.template insert_from_reg_no_vector<TM, TN>(threadRow * TM, threadCol * TN, c_reg);
}

int main() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    // impl::SharedMemoryManager smem_manager;
    int32_t offset = 0;
    impl::MatrixGPU<M, K, K, 1> A;
    offset += BM * K * sizeof(float);
    // impl::MatrixGPU<K, N, N, 1> B;
    impl::ColumnMajorMatrix<K, N> B(offset);

    offset += 32 * 32 * sizeof(float);
    impl::MatrixGPU<M, N, N, 1> C(offset);

    A.ascending();
    B.ascending();
    C.vvals(0.0f);

    float alpha = 1.0f;
    float beta = 0.0f;

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    auto func = [&]() {
        sgemmVectorize<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A.data, B.data, beta, C.data);
    };

    double time = benchmark::benchmark(10, 1, func, 2);
    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "GFLOPS: " << static_cast<double>(int64_t(2) * M * N * K) / (time * 1e9) << std::endl;

    impl::MatrixGPU<M, N, N, 1> C_gern(offset);
    auto func_gern = [&]() {
        gernel_mine<BM, BN, BK, TM, TN, impl::MatrixGPU<M, K, K, 1>, impl::ColumnMajorMatrix<K, N>, impl::MatrixGPU<M, N, N, 1>>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, C_gern);
    };

    double time_gern = benchmark::benchmark(10, 1, func_gern, 2);
    std::cout << "Time: " << time_gern << " ms" << std::endl;
    std::cout << "GFLOPS: " << static_cast<double>(int64_t(2) * M * N * K) / (time_gern * 1e9) << std::endl;

    auto C_gern_cpu = C_gern.get();
    auto C_cpu = C.get();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            assert(C_gern_cpu(i, j) == C_cpu(i, j));
        }
    }
}