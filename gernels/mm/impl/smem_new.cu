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

template<const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    // the output block that we want to compute in this threadblock
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                     // row=cRow, col=0
    B += cCol * BLOCKSIZE;                         // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;  // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * N + threadCol] = tmp;
}

template<const int BLOCKSIZE, const int M, const int N, const int K>
__global__ void gernel_mine(impl::MatrixGPU<M, K, K, 1> A_ds,
                            impl::ColumnMajorMatrix<K, N> B_ds,
                            impl::MatrixGPU<M, N, N, 1> C_ds) {
    // the output block that we want to compute in this threadblock
    const uint cRow = blockIdx.x * BLOCKSIZE;
    const uint cCol = blockIdx.y * BLOCKSIZE;

    impl::SharedMemoryManager smem_manager;
    // float *As = (float *)smem_manager.malloc(BLOCKSIZE * BLOCKSIZE * sizeof(float));
    // float *Bs = (float *)smem_manager.malloc(BLOCKSIZE * BLOCKSIZE * sizeof(float));

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    auto a_q = A_ds.template query_global_2_global<BLOCKSIZE, K>(cRow, 0);
    auto b_q = B_ds.template query_global_2_global<K, BLOCKSIZE>(0, cCol);
    auto c_q = C_ds.template query_global_2_global<BLOCKSIZE, BLOCKSIZE>(cRow, cCol);

    auto c_q_req = C_ds.template query_2_reg_no_vector_zero<1, 1>(cRow + threadRow, cCol + threadCol);

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {

        float *As = (float *)smem_manager.malloc(BLOCKSIZE * BLOCKSIZE * sizeof(float));
        float *Bs = (float *)smem_manager.malloc(BLOCKSIZE * BLOCKSIZE * sizeof(float));

        auto a_s_mine = a_q.template query_global_2_shared<BLOCKSIZE, BLOCKSIZE, BLOCKSIZE * BLOCKSIZE>(0, bkIdx, As);
        auto b_s_mine = b_q.template query_global_2_shared<BLOCKSIZE, BLOCKSIZE, BLOCKSIZE * BLOCKSIZE>(bkIdx, 0, Bs);

        __syncthreads();

        auto a_reg = a_s_mine.template query_global_2_global<1, BLOCKSIZE>(threadRow, 0);
        auto b_reg = b_s_mine.template query_global_2_global<BLOCKSIZE, 1>(0, threadCol);

        matrix_multiply<BLOCKSIZE>(a_reg, b_reg, c_q_req);

        __syncthreads();

        smem_manager.free();
    }

    C_ds.template insert_2_reg_no_vector<1, 1>(cRow + threadRow, cCol + threadCol, c_q_req);
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    impl::SharedMemoryManager smem_manager;

    impl::MatrixGPU<M, K, K, 1> A(&smem_manager);
    // impl::MatrixGPU<K, N, N, 1> B;
    impl::ColumnMajorMatrix<K, N> B(&smem_manager);
    impl::MatrixGPU<M, N, N, 1> C;

    A.ascending();
    B.ascending();
    C.vvals(0.0f);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);

    auto func = [&]() {
        sgemm_shared_mem_block<32><<<gridDim, blockDim>>>(M, N, K, 1.0f, A.data, B.data, 0.0f, C.data);
    };

    double time = benchmark::benchmark(10, 1, func, 2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "GFLOPS: " << static_cast<double>(int64_t(2) * M * N * K) / (time * 1e9) << std::endl;
    impl::MatrixGPU<M, N, N, 1> C_mine;
    C_mine.vvals(0.0f);

    auto gern_sp = gernel_mine<32, M, N, K>;
    int64_t smem_size = 32 * 32 * 8 * 10;

    cudaFuncSetAttribute(gern_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    auto func_mine = [&]() {
        gern_sp<<<gridDim, blockDim, smem_size>>>(A, B, C_mine);
    };

    double time_mine = benchmark::benchmark(10, 1, func_mine, 2);
    std::cout << "Time mine: " << time_mine << " ms" << std::endl;
    std::cout << "GFLOPS mine: " << static_cast<double>(int64_t(2) * M * N * K) / (time_mine * 1e9) << std::endl;

    auto C_cpu = C.get();
    auto C_cpu_mine = C_mine.get();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            assert(C_cpu(i, j) == C_cpu_mine(i, j));
        }
    }

    A.destroy();
    B.destroy();
    C.destroy();
}