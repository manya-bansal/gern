#pragma once

#include "gpu-array.h"
#include <cuda_runtime.h>

namespace library {
namespace impl {

template<int Size>
__device__ ArrayStaticGPU<Size> allocate_local() {
    return ArrayStaticGPU<Size>();
}

__device__ inline void add_1(ArrayGPU a, ArrayGPU b) {
    for (int64_t i = 0; i < a.size; i++) {
        b.data[i] = a.data[i] + 1;
    }
}

template<int Size>
__device__ inline void add_1(ArrayStaticGPU<Size> &a, ArrayStaticGPU<Size> &b) {
    for (int64_t i = 0; i < Size; i++) {
        b.data[i] = a.data[i] + 1;
    }
}

__device__ inline void add_1_thread(ArrayGPU a, ArrayGPU b) {
    int x = threadIdx.x;
    if (x > a.size)
        return;
    b.data[x] = a.data[x] + 1;
}

}  // namespace impl
}  // namespace gern