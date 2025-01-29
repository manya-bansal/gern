#pragma once

#include "cpu-matrix.h"

#include <cstring>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace gern {
namespace impl {

/**
 * @brief An MatrixGPU class to test out Gern's
 * codegen facilities.
 *
 */

class MatrixGPU {
public:
    MatrixGPU(int64_t row, int64_t col, int64_t lda)
        : row(row), col(col), lda(lda) {
        cudaMalloc(&data, lda * col * sizeof(float));
    }

    __device__ MatrixGPU(float *data, int64_t row, int64_t col, int64_t lda)
        : data(data), row(row), col(col), lda(lda) {
    }

    __device__ MatrixGPU query(int64_t x, int64_t y, int64_t l_x, int64_t l_y) {
        return MatrixGPU(data + (x * lda + y), l_x, l_y, lda);
    }

    MatrixCPU get() {
        MatrixCPU cpu(row, col, lda);
        cudaMemcpy(cpu.data, data, lda * col * sizeof(float), cudaMemcpyDeviceToHost);
        return cpu;
    }

    void destroy() {
        cudaFree(data);
    }

    void vvals(float f) {
        MatrixCPU tmp(row, col, lda);
        tmp.vvals(f);
        cudaMemcpy(data, tmp.data, lda * col * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    float *data;
    int64_t row;
    int64_t col;
    int64_t lda;
};

__device__ inline void add(MatrixGPU a, MatrixGPU b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);
        for (int64_t j = 0; j < a.col; j++) {
            b_data[j] = a_data[j] + 1;
        }
    }
}

}  // namespace impl
}  // namespace gern
