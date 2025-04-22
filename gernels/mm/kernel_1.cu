#include "../benchmark.h"
#include "../current_path.h"
#include "compose/runner.h"
#include "gern_annot/adt.h"
#include "gern_annot/functions.h"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include "mm_helpers.h"
#include "schedules.h"
#include <assert.h>
#include <iostream>

#include "kernel_1.h"

using namespace gern;

template<int64_t block_x, int64_t block_y, int64_t k_dim, int64_t thread_x>
__global__ void function_31(impl::MatrixGPU<M_CONST, N_CONST, K_CONST, 1> A, impl::MatrixGPU<M_CONST, N_CONST, K_CONST, 1> B, impl::MatrixGPU<M_CONST, N_CONST, K_CONST, 1> C) {
    int64_t _gern_i_1_7_13_19_25 = ((((blockIdx.x / 1) % (((block_x + (C.row - 0)) - 1) / block_x)) * block_x) + 0);
    int64_t _gern_j_2_8_14_20 = ((((blockIdx.y / 1) % (((block_y + (C.col - 0)) - 1) / block_y)) * block_y) + 0);
    int64_t _gern_i_1_7_13 = ((((threadIdx.x / 1) % (((thread_x + (block_x - 0)) - 1) / thread_x)) * thread_x) + 0);
    int64_t _gern_j_2_8 = ((((threadIdx.y / 1) % (((thread_x + (block_y - 0)) - 1) / thread_x)) * thread_x) + 0);

    auto _query_C_32 = C.template query_global_2_global<thread_x, thread_x>((_gern_i_1_7_13 + _gern_i_1_7_13_19_25), (_gern_j_2_8 + (_gern_j_2_8_14_20 + 0)));

    auto _query_A_33 = A.template query_global_2_global<thread_x, k_dim>((_gern_i_1_7_13 + _gern_i_1_7_13_19_25), 0);

    auto _query_B_34 = B.template query_global_2_global<k_dim, thread_x>(0, (_gern_j_2_8 + (_gern_j_2_8_14_20 + 0)));

    matrix_multiply<k_dim>(_query_A_33, _query_B_34, _query_C_32);
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        // C[x * N + y] = alpha * tmp + beta * C[x * N + y];
        C[x * N + y] = tmp;
    }
}

int main() {

    constexpr int64_t m = M_CONST;
    constexpr int64_t n = N_CONST;
    constexpr int64_t k = K_CONST;

    constexpr int64_t block_size = 1;

    using AImpl = impl::MatrixGPU<m, k, k, block_size>;
    using BImpl = impl::MatrixGPU<k, n, n, block_size>;
    using CImpl = impl::MatrixGPU<m, n, n, block_size>;

    Composable program = schedules::kernel_1(m, n, k, block_size);
    Runner runner = mm_helpers::runner(program, "kernel_1.cu");

    AImpl A;
    A.ascending();
    BImpl B;
    B.ascending();
    CImpl C;
    C.vvals(0.0f);

    // Make sure it's correct!
    // mm_helpers::evalute_and_check(runner, A, B, C);

    dim3 grid(m / 32, n / 32, 1);
    dim3 thrds(32, 32, 1);

    // Now, let's benchmark it!
    auto func = [&]() {
        // runner.evaluate({{"A", &A},
        //                  {"B", &B},
        //                  {"C", &C}});
        function_31<32, 32, k, 1><<<grid, thrds>>>(A, B, C);
    };

    double time = benchmark::benchmark(10, 1, func, 2);
    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "GFLOPS: " << mm_helpers::gflops(m, n, k, time) << std::endl;
    std::cout << "% of peak " << mm_helpers::gflops(m, n, k, time) / (44 * 10) << std::endl;

    auto blog = [&]() {
        sgemm_naive<<<grid, thrds>>>(m, n, k, 1.0f, A.data, B.data, 0.0f, C.data);
    };
    double time_naive = benchmark::benchmark(10, 1, blog, 2);
    std::cout << "Time naive: " << time_naive << " ms" << std::endl;
    std::cout << "GFLOPS naive: " << mm_helpers::gflops(m, n, k, time_naive) << std::endl;

    // Free everything.
    A.destroy();
    B.destroy();
    C.destroy();
}