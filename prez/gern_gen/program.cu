#include "/home/manya/gern/test/library/smem_allocator/sh_malloc.h"
#include "cassert"
#include "gpu-array.cu"
#include "gpu-array.h"
#include <cuda_runtime.h>

template<int64_t smem_size, int64_t tile, int64_t tile2>
__global__ void function_7(gern::impl::ArrayGPU a, gern::impl::ArrayGPU b) {

    init_shmem(smem_size);

    int64_t _gern_x_2_4_6 = ((blockIdx.x * tile) + 0);
    int64_t _gern_x_2_4 = ((threadIdx.x * tile2) + 0);
    int64_t _gern_x_2 = (_gern_x_2_4 + _gern_x_2_4_6);
    constexpr int64_t _gern_step_1 = tile2;

    auto _query_b_8 = b.query(_gern_x_2, _gern_step_1);

    auto _query_a_9 = a.query(_gern_x_2, _gern_step_1);

    gern::impl::add_1(_query_a_9, _query_b_8);
}

extern "C" {
void hook_function_7(void **args) {
    constexpr int64_t smem_size = 1024;
    constexpr int64_t tile = 5;
    constexpr int64_t tile2 = 1;
    gern::impl::ArrayGPU &a = *((gern::impl::ArrayGPU *)args[0]);
    gern::impl::ArrayGPU &b = *((gern::impl::ArrayGPU *)args[1]);
    dim3 grid_10 = dim3((((tile + (b.size - 0)) - 1) / tile), 1, 1);
    dim3 block_11 = dim3((((tile2 + (tile - 0)) - 1) / tile2), 1, 1);
    auto function_sp_12 = function_7<smem_size, tile, tile2>;
    cudaFuncSetAttribute(function_sp_12,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    function_sp_12<<<grid_10, block_11, smem_size>>>(a, b);
    ;
}
}
