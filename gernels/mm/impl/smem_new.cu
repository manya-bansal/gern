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

constexpr int dim = 256;

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
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}

template<int64_t block_x, int64_t block_y, int64_t k_dim, int64_t k_tiled, int64_t smem_size, int64_t thread_x, int64_t thread_y>
__global__ void function_73(impl::MatrixGPU<dim, dim, dim, 1> A, impl::ColumnMajorMatrix<dim, dim> B, impl::MatrixGPU<dim, dim, dim, 1> C) {

    int64_t _gern_i_1_7_13_19_25_31_37_43_49_55_61_67 = ((((blockIdx.x / 1) % (((block_x + (C.row - 0)) - 1) / block_x)) * block_x) + 0);
    auto _query_A_74 = A.template query_global_2_global<block_x, k_dim>(_gern_i_1_7_13_19_25_31_37_43_49_55_61_67, 0);

    int64_t _gern_j_2_8_14_20_26_32_38_44_50_56_62 = ((((blockIdx.y / 1) % (((block_y + (C.col - 0)) - 1) / block_y)) * block_y) + 0);
    auto _query_B_75 = B.template query_global_2_global<k_dim, block_y>(0, (_gern_j_2_8_14_20_26_32_38_44_50_56_62 + 0));

    auto _query_C_76 = C.template query_global_2_global<block_x, block_y>(_gern_i_1_7_13_19_25_31_37_43_49_55_61_67, (_gern_j_2_8_14_20_26_32_38_44_50_56_62 + 0));

    int64_t _gern_i_1_7_13_19_25 = ((((threadIdx.x / (1 * (((thread_y + (block_y - 0)) - 1) / thread_y))) % (((thread_x + (block_x - 0)) - 1) / thread_x)) * thread_x) + 0);
    int64_t _gern_j_2_8_14_20 = ((((threadIdx.x / 1) % (((thread_y + (block_y - 0)) - 1) / thread_y)) * thread_y) + 0);

    auto _query_C_81 = _query_C_76.template query_2_reg_no_vector_zero<thread_x, thread_y>((_gern_i_1_7_13_19_25_31_37_43_49_55_61_67 + _gern_i_1_7_13_19_25 + 0), (_gern_j_2_8_14_20_26_32_38_44_50_56_62 + _gern_j_2_8_14_20 + 0));

    for (int64_t _gern_k_3_9_15_21_27_33_39_45 = 0; (_gern_k_3_9_15_21_27_33_39_45 < k_dim); _gern_k_3_9_15_21_27_33_39_45 = (_gern_k_3_9_15_21_27_33_39_45 + k_tiled)) {
        auto _query_A_77 = _query_A_74.template query_global_2_shared_restrict<block_x, k_tiled>(0, (_gern_k_3_9_15_21_27_33_39_45 + 0));

        auto _query_B_78 = _query_B_75.template query_global_2_shared_restrict<k_tiled, block_y>((_gern_k_3_9_15_21_27_33_39_45 + 0), 0);

        auto _query_A_79 = _query_A_77.template query_global_2_global_sync<thread_x, k_tiled>((_gern_i_1_7_13_19_25 + 0), 0);

        auto _query_B_80 = _query_B_78.template query_global_2_global<k_tiled, thread_y>(0, (_gern_j_2_8_14_20 + 0));

        // auto _query_C_81 = _query_C_76.template query_global_2_global<thread_x, thread_y>((_gern_i_1_7_13_19_25 + 0), (_gern_j_2_8_14_20 + 0));

        matrix_multiply_sync<k_tiled>(_query_A_79, _query_B_80, _query_C_81);
    }

    C.template insert_2_reg_no_vector<thread_x, thread_y>((_gern_i_1_7_13_19_25_31_37_43_49_55_61_67 + _gern_i_1_7_13_19_25 + 0), (_gern_j_2_8_14_20_26_32_38_44_50_56_62 + _gern_j_2_8_14_20 + 0), _query_C_81);
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

        auto a_s_mine = a_q.template query_global_2_shared_restrict<BLOCKSIZE, BLOCKSIZE>(0, bkIdx);
        auto b_s_mine = b_q.template query_global_2_shared_restrict<BLOCKSIZE, BLOCKSIZE>(bkIdx, 0);

        // __syncthreads();

        auto a_reg = a_s_mine.template query_global_2_global_sync<1, BLOCKSIZE>(threadRow, 0);
        auto b_reg = b_s_mine.template query_global_2_global<BLOCKSIZE, 1>(0, threadCol);

        matrix_multiply_sync<BLOCKSIZE>(a_reg, b_reg, c_q_req);

        // __syncthreads();
    }

    C_ds.template insert_2_reg_no_vector<1, 1>(cRow + threadRow, cCol + threadCol, c_q_req);
}

int main() {
    constexpr int M = dim;
    constexpr int N = dim;
    constexpr int K = dim;

    // impl::SharedMemoryManager smem_manager;
    int32_t offset = 0;
    impl::MatrixGPU<M, K, K, 1> A;
    offset += 32 * 32 * sizeof(float);
    // impl::MatrixGPU<K, N, N, 1> B;
    impl::ColumnMajorMatrix<K, N> B(offset);

    offset += 32 * 32 * sizeof(float);
    impl::MatrixGPU<M, N, N, 1> C(offset);

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
    constexpr int64_t smem_size = 32 * 32 * 8 * 10;

    cudaFuncSetAttribute(gern_sp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    auto func_mine = [&]() {
        gern_sp<<<gridDim, blockDim, smem_size>>>(A, B, C_mine);
    };

    double time_mine = benchmark::benchmark(10, 1, func_mine, 2);
    std::cout << "Time mine: " << time_mine << " ms" << std::endl;
    std::cout << "GFLOPS mine: " << static_cast<double>(int64_t(2) * M * N * K) / (time_mine * 1e9) << std::endl;

    constexpr int64_t block_x = 32;
    constexpr int64_t block_y = 32;
    constexpr int64_t k_dim = dim;
    constexpr int64_t k_tiled = 32;
    constexpr int64_t thread_x = 1;
    constexpr int64_t thread_y = 1;
    dim3 grid_82 = dim3((1 * (((block_x + (C.row - 0)) - 1) / block_x)), (1 * (((block_y + (C.col - 0)) - 1) / block_y)), 1);
    dim3 block_83 = dim3(((1 * (((thread_y + (block_y - 0)) - 1) / thread_y)) * (((thread_x + (block_x - 0)) - 1) / thread_x)), 1, 1);
    auto function_sp_84 = function_73<block_x, block_y, k_dim, k_tiled, smem_size, thread_x, thread_y>;
    cudaFuncSetAttribute(function_sp_84, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    impl::MatrixGPU<M, N, N, 1> C_gern;
    C_gern.vvals(0.0f);

    auto func_gern = [&]() {
        function_sp_84<<<grid_82, block_83, smem_size>>>(A, B, C_gern);
    };

    double time_gern = benchmark::benchmark(10, 1, func_gern, 2);
    std::cout << "Time gern: " << time_gern << " ms" << std::endl;
    std::cout << "GFLOPS gern: " << static_cast<double>(int64_t(2) * M * N * K) / (time_gern * 1e9) << std::endl;

    auto C_cpu = C.get();
    auto C_cpu_mine = C_mine.get();
    auto C_cpu_gern = C_gern.get();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            assert(C_cpu(i, j) == C_cpu_mine(i, j));
            assert(C_cpu(i, j) == C_cpu_gern(i, j));
        }
    }

    A.destroy();
    B.destroy();
    C.destroy();
}