#include "cassert"
#include "impl/gpu-matrix-const.h"
#include "impl/impl.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

constexpr int64_t row_val_set = 4488 * 8;

template<int64_t col, int64_t row, int64_t stride>
__global__ void function_19(impl::MatrixGPU<row_val_set + 4, row_val_set + 4, row_val_set + 4, 1> input, impl::MatrixGPU<row_val_set, row_val_set, row_val_set, 1> output) {
    int64_t _gern_y_4_12_17 = ((threadIdx.x * col) + 0);
    int64_t _gern_x_3_11 = ((threadIdx.y * row) + 0);
    int64_t _gern_y_4 = _gern_y_4_12_17;
    constexpr int64_t _gern_l_y_2 = col;
    int64_t _gern_x_3 = _gern_x_3_11;
    constexpr int64_t _gern_l_x_1 = row;

    int64_t _gern_x_7 = _gern_x_3;
    int64_t _gern_y_8 = _gern_y_4;
    constexpr int64_t _gern_l_x_5 = ((_gern_l_x_1 + stride) - 1);
    constexpr int64_t _gern_l_y_6 = _gern_l_y_2;
    auto _query_output_20 = output.template query<_gern_l_x_1, _gern_l_y_2>(_gern_x_3, _gern_y_4);

    auto temp = impl::allocate_static<_gern_l_x_5, _gern_l_y_6>();

    auto _query_input_21 = input.template query<_gern_l_x_5, ((_gern_l_y_6 + stride) - 1)>(_gern_x_7, _gern_y_8);

    blur_x<stride>(_query_input_21, temp);

    blur_y<stride>(temp, _query_output_20);

    output.template insert(_gern_x_3, _gern_y_4, _query_output_20);
}

int main() {
    constexpr int64_t row_val = row_val_set;
    constexpr int64_t col_val = row_val_set;

    constexpr int64_t size_y_chain = 1;
    constexpr int64_t size_x_chain = 1;
    constexpr int64_t stride_val = 3;
    constexpr int64_t block_size = 1;

    // This needs to be a multiple of 4 to work with vectorization.
    constexpr int64_t row_val_in = row_val + stride_val + (size_y_chain - 2) + 2;  // Want to pad by a multiple of four.
    constexpr int64_t col_val_in = col_val + stride_val + (size_x_chain - 2) + 2;  // Want to pad by a multiple of four.;

    using InputType = impl::MatrixGPU<row_val_in, col_val_in, col_val_in, block_size>;
    using OutputType = impl::MatrixGPU<row_val, col_val, col_val, block_size>;

    InputType in;
    in.ascending();
    OutputType out;
    out.vvals(0.0f);

    constexpr int64_t col = 4;
    constexpr int64_t row = 4;
    constexpr int64_t stride = 3;

    dim3 grid_22 = dim3(1, 1, 1);
    dim3 block_23 = dim3((((col + (out.col - 0)) - 1) / col), (((row + (out.row - 0)) - 1) / row), 1);

    auto start = std::chrono::high_resolution_clock::now();
    function_19<col, row, stride><<<grid_22, block_23>>>(in, out);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << "s" << std::endl;

    auto start_2 = std::chrono::high_resolution_clock::now();
    function_19<col, row, stride><<<grid_22, block_23>>>(in, out);
    cudaDeviceSynchronize();
    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_2 = end_2 - start_2;
    std::cout << "Time taken: " << elapsed_2.count() << "s" << std::endl;

    auto start_3 = std::chrono::high_resolution_clock::now();
    function_19<col, row, stride><<<grid_22, block_23>>>(in, out);
    cudaDeviceSynchronize();
    auto end_3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_3 = end_3 - start_3;
    std::cout << "Time taken: " << elapsed_3.count() << "s" << std::endl;
}
