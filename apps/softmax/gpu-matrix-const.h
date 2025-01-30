#pragma once

#include "cpu-matrix.h"
#include "library/array/impl/gpu-array.h"

#include <cstring>
#include <cuda_runtime.h>
#include <stdlib.h>

template<int Size>
struct holder {
    float val[Size];
    static constexpr int size = Size;
};

template<int num_row, int num_col>
struct reg_array_internal {
    float4 array[num_row * num_col];
    static constexpr int rows = num_row;
    static constexpr int cols_by_4 = num_col;
};

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif
constexpr int URF{UNROLL_FACTOR};

#define CEILING(x, y) (((x) + (y) - 1) / (y))

namespace gern {
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

template<int64_t Row, int64_t Col, int64_t LDA>
class MatrixGPUConst {
public:
    MatrixGPUConst() {
        cudaMalloc(&data, lda * col * sizeof(float));
    }

    template<int64_t num_row, int64_t num_col, int64_t stride>
    __device__ reg_array_internal<num_row, CEILING((Col / 4), stride)> query_obj(int64_t x, int64_t y) {
        reg_array_internal<num_row, CEILING((Col / 4), stride)> matrix;
        auto &reg_array = matrix.array;
        float4 *val_ptr = reinterpret_cast<float4 *>(&data[x * row + y * 4]);

        int index = 0;
        for (int m = 0; m < num_row; m++) {
#pragma unroll URF
            for (int i = 0; i < num_col; i++) {
                float4 val = val_ptr[index];
                reg_array[i] = val;
                index += stride;
            }
        }
        return matrix;
    }

    template<int64_t NumRow, int64_t NumCol>
    __device__ MatrixStaticGPU<NumRow, NumCol> insert(int64_t x,
                                                      int64_t y,
                                                      MatrixStaticGPU<NumRow, NumCol> matrix) {

        float *static_matrix = matrix.data;
        float *global_matrix = data + (x * lda) + y;
        for (int64_t i = 0; i < NumRow; i++) {
            for (int64_t j = 0; j < NumCol; j++) {
                global_matrix[j] = static_matrix[j];
            }
            static_matrix += NumCol;
            global_matrix += lda;
        }
        return matrix;
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

    void ascending() {
        ArrayCPU tmp(lda * col);
        tmp.ascending();
        cudaMemcpy(data, tmp.data, lda * col * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    float *data;
    static constexpr int64_t row = Row;
    static constexpr int64_t col = Col;
    static constexpr int64_t lda = LDA;
};

}  // namespace impl
}  // namespace gern
