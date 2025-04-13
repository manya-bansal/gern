#pragma once

#include <cstdlib>
#include <cstring>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// This is an atomic sum.
template<int Size>
struct ArrayStaticGPU {
    float data[Size] = {0};
    static constexpr int64_t size = Size;
};

class ArrayCPU {
public:
    ArrayCPU(float *data, int64_t size)
        : data(data), size(size) {
    }
    ArrayCPU(int64_t size)
        : ArrayCPU((float *)calloc(size, sizeof(float)), size) {
    }
    static ArrayCPU allocate(int64_t start, int64_t len) {
        (void)start;
        return ArrayCPU(len);
    }
    ArrayCPU query(int64_t start, int64_t len) {
        return ArrayCPU(data + start, len);
    }
    void insert(int64_t start, int64_t len, ArrayCPU to_insert) {
        std::memcpy(data + start, to_insert.data, len);
    }

    void destroy() {
        free(data);
    }

    void vvals(float f) {
        for (int64_t i = 0; i < size; i++) {
            data[i] = f;
        }
    }

    void ascending() {
        for (int64_t i = 0; i < size; i++) {
            data[i] = (float)i;
        }
    }

    float *data;
    int64_t size;
};

template<int Size>
__device__ ArrayStaticGPU<Size> allocate_local() {
    return ArrayStaticGPU<Size>();
}

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
