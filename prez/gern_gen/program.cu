#include "cassert"
#include "gpu-array.cu"
#include "gpu-array.h"
#include <cuda_runtime.h>

__global__ void function_3(library::impl::ArrayGPU a,
                           library::impl::ArrayGPU b) {

  library::impl::add_1(a, b);
}

extern "C" {
void hook_function_3(void **args) {
  library::impl::ArrayGPU &a = *((library::impl::ArrayGPU *)args[0]);
  library::impl::ArrayGPU &b = *((library::impl::ArrayGPU *)args[1]);
  dim3 grid_4 = dim3(1, 1, 1);
  dim3 block_5 = dim3(1, 1, 1);
  auto function_sp_6 = function_3;
  function_sp_6<<<grid_4, block_5>>>(a, b);
  ;
}
}
