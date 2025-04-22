#include "../benchmark.h"
#include "../current_path.h"
#include "compose/runner.h"
#include "float-error.h"
#include "gern_annot/adt.h"
#include "gern_annot/functions.h"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include "mm_helpers.h"
#include <assert.h>
#include <iostream>

#include "kernel_2.h"

using namespace gern;

template<int64_t block_x, int64_t block_y, int64_t k_dim, int64_t thread_x, int64_t dim>
__global__ void function_31(impl::MatrixGPU<dim, dim, dim, 1> A, impl::MatrixGPU<dim, dim, dim, 1> B, impl::MatrixGPU<dim, dim, dim, 1> C) {

    int64_t _gern_i_1_7_13_19_25 = ((((blockIdx.y / 1) % (((block_x + (C.row - 0)) - 1) / block_x)) * block_x) + 0);
    int64_t _gern_j_2_8_14_20 = ((((blockIdx.x / 1) % (((block_y + (C.col - 0)) - 1) / block_y)) * block_y) + 0);
    int64_t _gern_i_1_7_13 = ((((threadIdx.x / (1 * (((thread_x + (block_y - 0)) - 1) / thread_x))) % (((thread_x + (block_x - 0)) - 1) / thread_x)) * thread_x) + 0);
    auto _query_A_33 = A.template query_global_2_global<thread_x, k_dim>((_gern_i_1_7_13 + _gern_i_1_7_13_19_25), 0);

    int64_t _gern_j_2_8 = ((((threadIdx.x / 1) % (((thread_x + (block_y - 0)) - 1) / thread_x)) * thread_x) + 0);
    auto _query_C_32 = C.template query_global_2_global<thread_x, thread_x>((_gern_i_1_7_13 + _gern_i_1_7_13_19_25), (_gern_j_2_8 + (_gern_j_2_8_14_20 + 0)));

    auto _query_B_34 = B.template query_global_2_global<k_dim, thread_x>(0, (_gern_j_2_8 + (_gern_j_2_8_14_20 + 0)));

    matrix_multiply<k_dim>(_query_A_33, _query_B_34, _query_C_32);
}

template<const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // if statement is necessary to make things work under tile quantization
    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        // C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
        C[cRow * N + cCol] = tmp;
    }
}

int main() {

    constexpr int64_t m = M_CONST;
    constexpr int64_t n = N_CONST;
    constexpr int64_t k = K_CONST;
    constexpr int64_t block_size = 1;

    using AType = annot::MatrixGlobalToGlobal;
    using BType = annot::MatrixGlobalToGlobal;
    using CType = annot::MatrixGlobalToGlobal;

    using AImpl = impl::MatrixGPU<m, k, k, block_size>;
    using BImpl = impl::MatrixGPU<k, n, n, block_size>;
    using CImpl = impl::MatrixGPU<m, n, n, block_size>;

    auto A_DS = AbstractDataTypePtr(new const AType("A", m, k, block_size, false));
    auto B_DS = AbstractDataTypePtr(new const BType("B", k, n, block_size, false));
    auto C_DS = AbstractDataTypePtr(new const CType("C", m, n, block_size, false));

    Variable k_dim("k_dim");
    Variable block_x("block_x");
    Variable block_y("block_y");
    Variable thread_x("thread_x");
    Variable thread_y("thread_y");

    block_x = block_x.bind(32);   // 8 elements per block_x
    block_y = block_y.bind(32);   // 8 elements per block_y
    thread_x = thread_x.bind(1);  // 1 element per thread_x
    thread_y = thread_y.bind(1);  // 1 element per thread_y
    k_dim = k_dim.bind(k);

    annot::MatrixMultiply mm(A_DS, B_DS, C_DS);
    auto mm_sp = &mm[{
        {"k_dim", k_dim},
    }];

    // Distribute over blocks and threads trivially.
    // But does memory coalescing flipping col and row basically gives memory coalescing.
    Composable program = {
        Global(
            (Tile(C_DS["row"], block_x) || Grid::Unit::BLOCK_Y)(
                (Tile(C_DS["col"], block_y) || Grid::Unit::BLOCK_X)(
                    (Tile(C_DS["row"], thread_x) || Grid::Unit::THREAD_X)(
                        (Tile(C_DS["col"], thread_x) || Grid::Unit::THREAD_X)(
                            (*mm_sp)(A_DS, B_DS, C_DS)))))),
    };

    Runner runner = mm_helpers::runner(program, "kernel_2.cu");

    AImpl A;
    A.ascending();
    BImpl B;
    B.ascending();
    CImpl C;
    C.vvals(0.0f);

    // mm_helpers::evalute_and_check(runner, A, B, C);

    dim3 grid(m / 32, n / 32, 1);
    dim3 block(32 * 32);

    // Set up all the values.
    auto func = [&]() {
        // runner.evaluate({{A_DS.getName(), &A},
        //                  {B_DS.getName(), &B},
        //                  {C_DS.getName(), &C}});
        function_31<32, 32, k, 1><<<grid, block>>>(A, B, C);
    };

    double time = benchmark::benchmark(10, 1, func, 2);
    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "GFLOPS: " << mm_helpers::gflops(m, n, k, time) << std::endl;
    std::cout << "% of peak " << mm_helpers::gflops(m, n, k, time) / (44 * 10) << std::endl;

    auto func2 = [&]() {
        sgemm_global_mem_coalesce<32><<<grid, block>>>(m, n, k, 1.0f, A.data, B.data, 0.0f, C.data);
    };
    double time2 = benchmark::benchmark(10, 1, func2, 2);
    std::cout << "Time: " << time2 << " ms" << std::endl;
    std::cout << "GFLOPS: " << mm_helpers::gflops(m, n, k, time2) << std::endl;
    std::cout << "% of peak " << mm_helpers::gflops(m, n, k, time2) / (44 * 10) << std::endl;

    A.destroy();
    B.destroy();
    C.destroy();
}