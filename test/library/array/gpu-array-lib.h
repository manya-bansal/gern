#pragma once

#include "cpu-array-lib.h"

#include <cuda_runtime.h>
#include <stdlib.h>

namespace gern {
namespace lib {

class TestArrayGPU {
public:
    TestArrayGPU(int size)
        : size(size) {
        cudaMalloc(&data, size * sizeof(float));
    }
    __device__ TestArrayGPU(float *data, int size)
        : data(data), size(size) {
    }

    __device__ TestArrayGPU query(int start, int len) {
        return TestArrayGPU(data + start, len);
    }

    void vvals(float v) {
        TestArray tmp(size);
        tmp.vvals(v);
        cudaMemcpy(data, tmp.data, size * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    TestArray get() {
        TestArray tmp(size);
        cudaMemcpy(tmp.data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
        return tmp;
    }

    void destroy() {
        cudaFree(data);
    }

    float *data;
    int size;
};

__device__ inline void add(TestArrayGPU a, TestArrayGPU b) {
    for (int64_t i = 0; i < a.size; i++) {
        b.data[i] += a.data[i];
    }
}

}  // namespace lib
}  // namespace gern
