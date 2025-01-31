#pragma once

#include "cpu-array.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

namespace gern {
namespace impl {

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<int Size>
struct ArrayStaticGPU {
    float data[Size] = {0};
};

class ArrayGPU {
public:
    ArrayGPU(int64_t size)
        : size(size) {
        gpuErrchk(cudaMalloc(&data, size * sizeof(float)));
    }
    __device__ ArrayGPU(float *data, int64_t size)
        : data(data), size(size) {
    }

    __device__ ArrayGPU query(int64_t start, int64_t len) {
        return ArrayGPU(data + start, len);
    }

    template<int Size>
    __device__ inline ArrayStaticGPU<Size> query(int start) {
        ArrayStaticGPU<Size> arr;
        for (int i = 0; i < Size; i++) {
            arr.data[i] = data[start + i];
        }
        return arr;
    }

    template<int Size>
    __device__ inline void insert_array(int start, ArrayStaticGPU<Size> &a) {
        for (int i = 0; i < Size; i++) {
            data[start + i] = a.data[i];
        }
    }

    void vvals(float v) {
        ArrayCPU tmp(size);
        tmp.vvals(v);
        cudaMemcpy(data, tmp.data, size * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    void ascending() {
        ArrayCPU tmp(size);
        tmp.ascending();
        cudaMemcpy(data, tmp.data, size * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    ArrayCPU get() {
        ArrayCPU cpu(size);
        cudaMemcpy(cpu.data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
        return cpu;
    }

    void destroy() {
        cudaFree(data);
    }

    __device__ void dummy() const {
    }

    float *data;
    int64_t size;
};

}  // namespace impl
}  // namespace gern
