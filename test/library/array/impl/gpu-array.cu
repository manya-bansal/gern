#pragma once

#include "gpu-array.h"
#include <cuda_runtime.h>

namespace gern {
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

template<typename T1, typename T2>
__device__ inline void reduction(T1 a, T2 b, int64_t k) {
    for (int64_t i = 0; i < b.size; i++) {
        for (int64_t j = 0; j < k; j++) {
            b.data[i] += a.data[j];
        }
    }
}

}  // namespace impl
}  // namespace gern