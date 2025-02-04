#include "cassert"
#include "impl/gpu-matrix-const.h"
#include "impl/impl.h"
#include <cuda_runtime.h>

template<int64_t col, int64_t col_val, int64_t row, int64_t stride_val>
__global__ void function_39(impl::MatrixGPU<16384, 16384, 16384, 1024> input, impl::MatrixGPU<16384, 16384, 16384, 1024> output) {

    int64_t _gern_x_3_24_35 = ((blockIdx.x * row) + 0);
    for (int64_t _gern_y_4_26 = 0; (_gern_y_4_26 < output.col); _gern_y_4_26 = (_gern_y_4_26 + col_val)) {

        int64_t _gern_y_4 = _gern_y_4_26;
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
        auto _query_output_40 = output.template query<_gern_l_x_1, _gern_l_y_2>(_gern_x_3, _gern_y_4);

        auto max_row_out = impl::allocate_static_array<_gern_l_x_15>();

        auto _query_input_41 = input.template query<_gern_l_x_15, col>(_gern_x_16, 0);

        max_shuffle<stride_val>(max_row_out, _query_input_41);

        auto sub_temp = impl::allocate_static<_gern_l_x_11, _gern_l_y_12>();

        subtract_vec(max_row_out, _query_input_41, sub_temp);

        auto exp_temp = impl::allocate_static<_gern_l_x_7, _gern_l_y_8>();

        exp_matrix(sub_temp, exp_temp);

        auto sum_row_out = impl::allocate_static_array<_gern_l_x_5>();
        sum_row<stride_val>(sum_row_out, exp_temp);

        divide_vec(sum_row_out, exp_temp, _query_output_40);

        output.template insert(_gern_x_3, _gern_y_4, _query_output_40);
    }
}

extern "C" {
void hook_function_39(void **args) {
    constexpr int64_t col = 16384;
    constexpr int64_t col_val = 16384;
    constexpr int64_t row = 4;
    constexpr int64_t stride_val = 1024;
    impl::MatrixGPU<16384, 16384, 16384, 1024> &input = *((impl::MatrixGPU<16384, 16384, 16384, 1024> *)args[0]);
    impl::MatrixGPU<16384, 16384, 16384, 1024> &output = *((impl::MatrixGPU<16384, 16384, 16384, 1024> *)args[1]);
    dim3 grid_42 = dim3((((row + (output.row - 0)) - 1) / row), 1, 1);
    dim3 block_43 = dim3(1, stride_val, 1);
    function_39<col, col_val, row, stride_val><<<grid_42, block_43>>>(input, output);
    ;
}
}
