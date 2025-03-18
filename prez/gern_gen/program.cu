#include "cassert"
#include "gpu-array.cu"
#include "gpu-array.h"
#include <cuda_runtime.h>

template <int64_t tile, int64_t tile2>
__global__ void function_7(library::impl::ArrayGPU a,
                           library::impl::ArrayGPU b) {

  int64_t _gern_x_2_4_6 = ((blockIdx.x * tile) + 0);
  int64_t _gern_x_2_4 = ((threadIdx.x * tile2) + 0);
  int64_t _gern_x_2 = (_gern_x_2_4 + _gern_x_2_4_6);
  constexpr int64_t _gern_step_1 = tile2;

  auto _query_b_8 = b.query(_gern_x_2, _gern_step_1);

  auto _query_a_9 = a.query(_gern_x_2, _gern_step_1);

  library::impl::add_1(_query_a_9, _query_b_8);
}

extern "C" {
void hook_function_7(void **args) {
  constexpr int64_t tile = 5;
  constexpr int64_t tile2 = 1;
  library::impl::ArrayGPU &a = *((library::impl::ArrayGPU *)args[0]);
  library::impl::ArrayGPU &b = *((library::impl::ArrayGPU *)args[1]);
  dim3 grid_10 = dim3((((tile + (b.size - 0)) - 1) / tile), 1, 1);
  dim3 block_11 = dim3((((tile2 + (tile - 0)) - 1) / tile2), 1, 1);
  auto function_sp_12 = function_7<tile, tile2>;
  function_sp_12<<<grid_10, block_11>>>(a, b);
  ;
}
}
