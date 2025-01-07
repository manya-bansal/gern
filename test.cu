#include "gpu-matrix.h"
#include <cuda_runtime.h>

extern "C" {
__global__ void function_39(gern::impl::MatrixGPU input_con, gern::impl::MatrixGPU output_con, int64_t col, int64_t l_x, int64_t row, int64_t l_y) {
    int64_t x = ((blockIdx.x * l_x) + 0);
    int64_t y = ((blockIdx.y * l_y) + 0);
    gern::impl::MatrixGPU _query_input_con = input_con.query(x, y, l_x, l_y);
    gern::impl::MatrixGPU _query_output_con = output_con.query(x, y, l_x, l_y);
    gern::impl::add(_query_input_con, _query_output_con);
}

void hook_function_39(void **args) {
    gern::impl::MatrixGPU input_con = *((gern::impl::MatrixGPU *)args[0]);
    gern::impl::MatrixGPU output_con = *((gern::impl::MatrixGPU *)args[1]);
    int64_t col = *((int64_t *)args[2]);
    int64_t l_x = *((int64_t *)args[3]);
    int64_t row = *((int64_t *)args[4]);
    int64_t l_y = *((int64_t *)args[5]);
    dim3 __grid_dim__((((l_x + (row - 0)) - 1) / l_x), (((l_y + (col - 0)) - 1) / l_y), 1);
    dim3 __block_dim__(1, 1, 1);
    function_39<<<__grid_dim__, __block_dim__>>>(input_con, output_con, col, l_x, row, l_y);
}
}