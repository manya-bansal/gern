// From https:github.com/SzymonOzog/FastSoftmax.git

#include "benchmark.h"
#include "impl/gpu-matrix-const.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

template<int stride,
         typename T,
         typename T2>
__device__ void query(T a,
                      int x,
                      int y,
                      T2 &reg_array_big) {
    auto &reg_array = reg_array_big.array;
    float4 *val_ptr = reinterpret_cast<float4 *>(&a.data[x * a.row + y * 4]);

    constexpr int64_t num_row = reg_array_big.rows;
    constexpr int64_t num_col = reg_array_big.cols_by_4;

    int index = 0;
    for (int m = 0; m < num_row; m++) {
#pragma unroll URF
        for (int i = 0; i < num_col; i++) {
            float4 val = val_ptr[index];
            reg_array[i] = val;
            index += stride;
        }
    }
}

template<int stride,
         typename T1,
         typename T2>
__device__ void insert(T1 a,
                       int x,
                       int y,
                       T2 &reg_array_big) {
    auto &reg_array = reg_array_big.array;
    constexpr int64_t num_row = reg_array_big.rows;
    constexpr int64_t num_col = reg_array_big.cols_by_4;
    float4 *val_ptr = reinterpret_cast<float4 *>(&a.data[x * a.row + y * 4]);

    int index = 0;
    for (int m = 0; m < num_row; m++) {
#pragma unroll URF
        for (int i = 0; i < num_col; i++) {
            float4 val = reg_array[i];
            val_ptr[index] = val;
            index += stride;
        }
    }
}

template<int num_row, int num_col>
__device__ StaticMatrix<num_row, num_col> allocate_local() {
    return StaticMatrix<num_row, num_col>();
}

#include "impl/impl.h"

template<int tile_row,
         int tile_col,
         typename T>
__global__ void softmax_kernel_mine(T a,
                                    T b) {
    int x = blockIdx.x;
    int y = threadIdx.y;

    constexpr int64_t num_cols_in = CEILING((a.col / 4), tile_col);
    constexpr int64_t num_rows_in = tile_row;

    // if (y < h)
    // {
    StaticMatrix<num_rows_in, num_cols_in> reg_array_big;
    query<tile_col>(a, x, y, reg_array_big);
    holder<num_rows_in> hold;
    max_shuffle<tile_col>(hold, reg_array_big);
    subtract_vec(hold, reg_array_big, reg_array_big);
    exp_matrix(reg_array_big, reg_array_big);
    sum_row<tile_col>(hold, reg_array_big);
    divide_vec(hold, reg_array_big, reg_array_big);
    insert<tile_col>(b, x, y, reg_array_big);
    // }
}

template<int64_t col, int64_t col_val, int64_t row, int64_t stride_val>
__global__ void function_39(impl::MatrixGPU<16384, 16384, 16384, 1024> input,
                            impl::MatrixGPU<16384, 16384, 16384, 1024> output) {

    int64_t _gern_x_3_24_35 = ((blockIdx.x * row) + 0);
    // for (int64_t _gern_y_4_26 = 0; (_gern_y_4_26 < output.col); _gern_y_4_26 = (_gern_y_4_26 + col_val)) {

    int64_t _gern_y_4 = threadIdx.y;
    constexpr int64_t _gern_l_y_2 = col_val;
    int64_t _gern_x_3 = _gern_x_3_24_35;
    constexpr int64_t _gern_l_x_1 = row;

    int64_t _gern_x_6 = _gern_x_3;
    constexpr int64_t _gern_l_x_5 = _gern_l_x_1;
    int64_t _gern_x_9 = _gern_x_3;
    int64_t _gern_y_10 = _gern_y_4;
    constexpr int64_t _gern_l_x_7 = _gern_l_x_1;
    constexpr int64_t _gern_l_y_8 = _gern_l_y_2;
    int64_t _gern_x_13 = _gern_x_9;
    int64_t _gern_y_14 = _gern_y_10;
    constexpr int64_t _gern_l_x_11 = _gern_l_x_7;
    constexpr int64_t _gern_l_y_12 = _gern_l_y_8;
    int64_t _gern_x_16 = _gern_x_13;
    constexpr int64_t _gern_l_x_15 = _gern_l_x_11;

    auto _query_output_40 = output.query_new<_gern_l_x_1, _gern_l_y_2>(_gern_x_3, _gern_y_4);

    auto max_row_out = impl::allocate_static_array<_gern_l_x_15>();

    auto _query_input_41 = input.query_new<_gern_l_x_15, col>(_gern_x_16, 0);

    max_shuffle<stride_val>(max_row_out, _query_input_41);

    auto sub_temp = impl::allocate_static<_gern_l_x_11, _gern_l_y_12>();

    subtract_vec(max_row_out, _query_input_41, sub_temp);

    auto exp_temp = impl::allocate_static<_gern_l_x_7, _gern_l_y_8>();

    exp_matrix(sub_temp, exp_temp);

    auto sum_row_out = impl::allocate_static_array<_gern_l_x_5>();

    sum_row<stride_val>(sum_row_out, exp_temp);

    divide_vec(sum_row_out, exp_temp, _query_output_40);

    output.insert_new(_gern_x_3, _gern_y_4, _query_output_40);
    // }
}

template<int tile_row,
         int tile_col,
         typename T>
__global__ void softmax_kernel_gern_like(T a,
                                         T b) {
    int x = blockIdx.x;
    int y = threadIdx.y;

    constexpr int64_t num_cols_q = CEILING(a.row, tile_col);
    constexpr int64_t num_rows_in = tile_row;

    auto reg_array_big = a.template query_new<num_rows_in, num_cols_q>(x, y);
    auto output_query = b.template query_new<num_rows_in, num_cols_q>(x, y);

    auto max_row_out = impl::allocate_static_array<num_rows_in>();
    max_shuffle<tile_col>(max_row_out, reg_array_big);

    auto sub_temp = impl::allocate_static<num_rows_in, num_cols_q>();
    subtract_vec(max_row_out, reg_array_big, sub_temp);

    auto exp_temp = impl::allocate_static<num_rows_in, num_cols_q>();
    exp_matrix(sub_temp, exp_temp);

    auto sum_row_out = impl::allocate_static_array<num_rows_in>();
    sum_row<tile_col>(sum_row_out, exp_temp);

    divide_vec(sum_row_out, exp_temp, output_query);

    b.insert_new(x, y, output_query);
}

constexpr int warm_up_runs = 5;
constexpr int kernel_repeats = 5;

#ifndef WIDTH
#define WIDTH 16384
#endif

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 1024
#endif

#include "original.cuh"

int main() {
    constexpr int64_t h = WIDTH;
    constexpr int64_t w = WIDTH;
    constexpr int tile_col = BLOCK_DIM_Y;

    using MatrixType = impl::MatrixGPU<h, w, h, tile_col>;
    std::cout << WIDTH << std::endl;

    MatrixType in;
    in.ascending();
    MatrixType out;
    out.vvals(0.0f);

    dim3 block_size = dim3(1, BLOCK_DIM_Y, 1);
    dim3 grid_size = dim3(h, 1, 1);

    constexpr int tile_row = 1;

    cudaStream_t stream = NULL;

    softmax_kernel10<float><<<grid_size, block_size>>>(in.data, out.data, w, h);
    impl::MatrixCPU reference = out.get();
    out.vvals(0.0f);

    auto specialized = softmax_kernel_gern_like<tile_row, tile_col, MatrixType>;
    // auto specialized = function_39<w, w, 1, tile_col>;
    specialized<<<grid_size, block_size>>>(in, out);
    impl::MatrixCPU gern = out.get();

    for (int64_t i = 0; i < h * w; i++)
        assert(reference.data[i] == gern.data[i]);

    double time = benchmark::measure::execution(
        [&](cudaStream_t stream) {
            specialized<<<grid_size, block_size>>>(in, out);
        },
        warm_up_runs,
        kernel_repeats,
        stream);

    double gflops = sizeof(float) * h * w * 2 * 1e-9;
    std::cout << gflops / (time / 1000) << std::endl;

    // specialized = softmax_kernel_mine<tile_row, tile_col, MatrixType>;
    // double time = benchmark::measure::execution(
    //     [&](cudaStream_t stream) {
    //         specialized<<<grid_size, block_size>>>(in, out);
    //     },
    //     warm_up_runs,
    //     kernel_repeats,
    //     stream);
    // std::cout << gflops / (time / 1000) << std::endl;

    out.destroy();
    in.destroy();
    reference.destroy();
    gern.destroy();
}