#pragma once

#include "library/matrix/impl/cpu-matrix.h"

#include <cstring>
#include <cuda_runtime.h>
#include <stdlib.h>

template<int Size>
struct holder {
    float val[Size];
    static constexpr int size = Size;
};

template<int NumRow, int NumCol>
struct StaticMatrix {
    float4 array[NumRow * NumCol];
    static constexpr int rows = NumRow;
    static constexpr int cols_by_4 = NumCol;
};

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif
constexpr int URF{UNROLL_FACTOR};

#define CEILING(x, y) (((x) + (y) - 1) / (y))

namespace impl {

/**
 * @brief An MatrixGPU class to test out Gern's
 * codegen facilities.
 *
 */

template<int64_t NumRow, int64_t NumCol>
struct MatrixStaticGPU {
    float data[NumRow * NumCol];
};

template<int64_t Row, int64_t Col, int64_t LDA, int64_t stride>
class MatrixGPU {
public:
    MatrixGPU() {
        cudaMalloc(&data, lda * col * sizeof(float));
    }

    template<int64_t num_row, int64_t num_col>
    __device__ StaticMatrix<num_row, num_col / 4> query(int64_t x, int64_t y) {
        StaticMatrix<num_row, num_col / 4> matrix;
        auto &matrix_data = matrix.array;
        float4 *val_ptr = reinterpret_cast<float4 *>(&data[x * row + y * 4]);

        int index = 0;
        for (int m = 0; m < num_row; m++) {
#pragma unroll URF
            for (int i = 0; i < num_col / 4; i++) {
                float4 val = val_ptr[index];
                matrix_data[i] = val;
                index += stride;
            }
        }
        return matrix;
    }

    template<typename T2>
    __device__ void insert(int x,
                           int y,
                           T2 &reg_array_big) {
        auto &reg_array = reg_array_big.array;
        constexpr int64_t num_row = reg_array_big.rows;
        constexpr int64_t num_col = reg_array_big.cols_by_4;
        float4 *val_ptr = reinterpret_cast<float4 *>(&data[x * row + y * 4]);

        int index = 0;
        for (int m = 0; m < num_row; m++) {
#pragma unroll URF
            for (int i = 0; i < num_col; i++) {
                float4 val = reg_array[i];
                val_ptr[index] = val;
                index += stride;
            }
        }
    }

    gern::impl::MatrixCPU get() {
        gern::impl::MatrixCPU cpu(row, col, lda);
        cudaMemcpy(cpu.data, data, lda * col * sizeof(float), cudaMemcpyDeviceToHost);
        return cpu;
    }

    void destroy() {
        cudaFree(data);
    }

    void vvals(float f) {
        gern::impl::MatrixCPU tmp(row, col, lda);
        tmp.vvals(f);
        cudaMemcpy(data, tmp.data, lda * col * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    void ascending() {
        gern::impl::ArrayCPU tmp(lda * col);
        tmp.ascending();
        cudaMemcpy(data, tmp.data, lda * col * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    float *data;
    static constexpr int64_t row = Row;
    static constexpr int64_t col = Col;
    static constexpr int64_t lda = LDA;
};

template<int64_t num_row, int64_t num_col>
__device__ StaticMatrix<num_row, num_col / 4> allocate_static() {
    return StaticMatrix<num_row, num_col / 4>();
}

}  // namespace impl
