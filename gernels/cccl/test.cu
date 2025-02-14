#include "cassert"
#include "wrappers/reduce_wrappers.cuh"
#include <cuda_runtime.h>

#include "cassert"
#include "wrappers/reduce_wrappers.cuh"
#include <cuda_runtime.h>

// template<int64_t t1, int64_t var_thrds_per_blk>
// __global__ void function_8(ArrayGPU input, float *output) {

//     int64_t _gern_i_2_5 = ((blockIdx.x * t1) + 0);

//     int64_t _gern_i_2 = _gern_i_2_5;
//     constexpr int64_t _gern_k_1 = t1;

//     int64_t _gern_i_3 = _gern_i_2;
//     constexpr int64_t _gern_lx_4 = _gern_k_1;
//     auto temp = allocate_local<_gern_lx_4>();

//     auto _query_input_9 = input.query<(var_thrds_per_blk * _gern_lx_4)>(((_gern_i_3 * var_thrds_per_blk) * _gern_lx_4));

//     block_reduce_take2<var_thrds_per_blk>(temp, _query_input_9);

//     global_sum<_gern_k_1>(output, temp);
// }

// extern "C" {
// void hook_function_8(void **args) {
//     constexpr int64_t t1 = 1;
//     constexpr int64_t var_thrds_per_blk = 2;
//     ArrayGPU &input = *((ArrayGPU *)args[0]);
//     float *&output = *((float **)args[1]);
//     dim3 grid_10 = dim3((((t1 + (4 - 0)) - 1) / t1), 1, 1);
//     dim3 block_11 = dim3(var_thrds_per_blk, 1, 1);
//     function_8<t1, var_thrds_per_blk><<<grid_10, block_11>>>(input, output);
//     ;
// }
// }

template<int64_t t1, int64_t var_thrds_per_blk>
__global__ void function_8(ArrayGPU input, float *output) {

    int64_t _gern_i_1_5 = ((blockIdx.x * t1) + 0);

    int64_t _gern_i_1 = _gern_i_1_5;
    constexpr int64_t _gern_tk_2 = t1;

    int64_t _gern_i_3 = _gern_i_1;
    constexpr int64_t _gern_lx_4 = _gern_tk_2;
    auto temp = allocate_local<_gern_lx_4>();

    auto _query_input_9 = input.query<(var_thrds_per_blk * _gern_lx_4)>(((_gern_i_3 * var_thrds_per_blk) * _gern_lx_4));

    block_reduce_take2<var_thrds_per_blk>(temp, _query_input_9);

    global_sum<t1>(output, temp);
}

extern "C" {
void hook_function_8(void **args) {
    constexpr int64_t t1 = 1;
    constexpr int64_t var_thrds_per_blk = 2;
    ArrayGPU &input = *((ArrayGPU *)args[0]);
    float *&output = *((float **)args[1]);
    dim3 grid_10 = dim3((((t1 + (4 - 0)) - 1) / t1), 1, 1);
    dim3 block_11 = dim3(var_thrds_per_blk, 1, 1);
    function_8<t1, var_thrds_per_blk><<<grid_10, block_11>>>(input, output);
    ;
}
}

int main() {
    int size_of_output = 4;
    int thread_per_block = 2;

    ArrayGPU output_real(size_of_output);
    output_real.vvals(0.0f);
    ArrayGPU input_real(size_of_output * thread_per_block);
    input_real.ascending();

    void *args[2];
    args[0] = &input_real;
    args[1] = &output_real.data;

    hook_function_8(args);
    auto output_cpu = output_real.get();
    std::cout << "Output: " << output_cpu.data[0] << std::endl;
}