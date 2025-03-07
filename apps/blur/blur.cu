#include "cassert"
#include "impl/gpu-matrix-const.h"
#include "impl/impl.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#include "value.h"

constexpr int64_t row_val_set = 128 * 120;

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after " << cudaGetErrorString(err) << std::endl;
    }
}

template<int64_t col, int64_t col_inner, int64_t row, int64_t row_inner, int64_t stride>
__global__ void function_19(impl::MatrixGPU<row_val_set + 4, row_val_set + 4, row_val_set + 4, 1> input, impl::MatrixGPU<row_val_set, row_val_set, row_val_set, 1> output) {
    int64_t _gern_y_4_12_17_22_27 = ((blockIdx.x * col) + 0);
    int64_t _gern_x_3_11_16_21 = ((blockIdx.y * row) + 0);
    int64_t _gern_y_4_12_17 = ((threadIdx.x * col_inner) + 0);
    int64_t _gern_x_3_11 = ((threadIdx.y * row_inner) + 0);
    int64_t _gern_y_4 = (_gern_y_4_12_17 + _gern_y_4_12_17_22_27);
    constexpr int64_t _gern_l_y_2 = col_inner;
    int64_t _gern_x_3 = (_gern_x_3_11 + _gern_x_3_11_16_21);
    constexpr int64_t _gern_l_x_1 = row_inner;

    int64_t _gern_x_7 = _gern_x_3;
    int64_t _gern_y_8 = _gern_y_4;
    constexpr int64_t _gern_l_x_5 = ((_gern_l_x_1 + stride) - 1);
    constexpr int64_t _gern_l_y_6 = _gern_l_y_2;
    auto _query_output_30 = output.template query<_gern_l_x_1, _gern_l_y_2>(_gern_x_3, _gern_y_4);

    auto temp = impl::allocate_static<_gern_l_x_5, _gern_l_y_6>();

    auto _query_input_31 = input.template query<_gern_l_x_5, ((_gern_l_y_6 + stride) - 1)>(_gern_x_7, _gern_y_8);

    blur_x<stride>(_query_input_31, temp);

    blur_y<stride>(temp, _query_output_30);

    output.template insert(_gern_x_3, _gern_y_4, _query_output_30);
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

    constexpr int64_t col = 16;
    constexpr int64_t col_inner = 4;
    constexpr int64_t row = 16;
    constexpr int64_t row_inner = 4;
    constexpr int64_t stride = 3;

    dim3 grid_32 = dim3((((col + (out.col - 0)) - 1) / col), (((row + (out.row - 0)) - 1) / row), 1);
    dim3 block_33 = dim3((((col_inner + (col - 0)) - 1) / col_inner), (((row_inner + (row - 0)) - 1) / row_inner), 1);

    checkCudaError();

    for (int i = 0; i < 10; i++) {
        function_19<col, col_inner, row, row_inner, stride><<<grid_32, block_33>>>(in, out);
    }

    cudaDeviceSynchronize();
    // Start actual benchmarking
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++) {
        function_19<col, col_inner, row, row_inner, stride><<<grid_32, block_33>>>(in, out);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = (end - start);
    auto min = (duration.count() / 200) / 1e6;

    // double measure = row_val * col_val * 4 * 2;
    double measure = 6 * row_val * col_val;
    std::cout << measure / min << std::endl;
}
