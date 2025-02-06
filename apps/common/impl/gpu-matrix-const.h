#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

template<int Size>
struct holder {
    float val[Size];
    static constexpr int size = Size;
};

template<int Size>
struct StaticArray {
    float val[Size];
    static constexpr int size = Size;
};

template<int NumRow, int NumCol>
struct StaticMatrix {
    float4 array[NumRow * NumCol];
    static constexpr int rows = NumRow;
    static constexpr int cols_by_4 = NumCol;
    template<int num_row, int num_col>
    __device__ StaticMatrix<num_row, num_col / 4> query_new(int64_t x, int64_t y) {
        return *this;
    }
};

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif
constexpr int URF{UNROLL_FACTOR};

#define CEILING(x, y) (((x) + (y) - 1) / (y))

namespace impl {

class MatrixCPU {
public:
    MatrixCPU(float *data, int64_t row, int64_t col, int64_t lda)
        : data(data), row(row), col(col), lda(lda) {
    }

    MatrixCPU(int64_t row, int64_t col, int64_t lda)
        : MatrixCPU((float *)malloc(sizeof(float) * lda * row), row, col, lda) {
    }

    static MatrixCPU allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return MatrixCPU(l_x, l_y, l_x);
    }

    MatrixCPU query(int64_t x, int64_t y, int64_t l_x, int64_t l_y) {
        return MatrixCPU(data + (x * lda + y), l_x, l_y, lda);
    }

    void insert(int64_t x, int64_t y, int64_t l_x, int64_t l_y, MatrixCPU to_insert) {
        float *data_tmp;
        float *data_insert_tmp;
        for (int64_t i = 0; i < l_x; i++) {
            data_tmp = data + ((x + i) * lda) + y;
            data_insert_tmp = to_insert.data + (i * lda);
            for (int64_t j = 0; j < l_y; j++) {
                data_tmp[j] = data_insert_tmp[j];
            }
        }
    }

    void ascending() {
        for (int64_t i = 0; i < lda * row; i++) {
            data[i] = i / 10000;
        }
    }

    void destroy() {
        free(data);
    }

    void vvals(float f) {
        float *data_tmp;
        for (int64_t i = 0; i < row; i++) {
            data_tmp = data + (i * lda);
            for (int64_t j = 0; j < col; j++) {
                data_tmp[j] = f;
            }
        }
    }

    float *data;
    int64_t row;
    int64_t col;
    int64_t lda;
};

[[maybe_unused]] static std::ostream &operator<<(std::ostream &os, const MatrixCPU &m) {
    float *data_tmp;
    os << "[" << "\n";
    for (int64_t i = 0; i < m.row; i++) {
        data_tmp = m.data + (i * m.lda);
        for (int64_t j = 0; j < m.col; j++) {
            os << data_tmp[j] << " ";
        }
        os << "\n";
    }
    os << "]";
    return os;
}

template<int64_t Row, int64_t Col, int64_t LDA, bool RowMajor>
class MatrixGPU {
public:
    MatrixGPU() {
        cudaMalloc(&data, lda * row * sizeof(float));
    }

    MatrixCPU get() {
        MatrixCPU cpu(row, col, lda);
        cudaMemcpy(cpu.data, data, lda * row * sizeof(float), cudaMemcpyDeviceToHost);
        return cpu;
    }

    void destroy() {
        cudaFree(data);
    }

    void vvals(float f) {
        MatrixCPU tmp(row, col, lda);
        tmp.vvals(f);
        cudaMemcpy(data, tmp.data, lda * row * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    void ascending() {
        MatrixCPU tmp(row, col, lda);
        tmp.ascending();
        cudaMemcpy(data, tmp.data, lda * row * sizeof(float), cudaMemcpyHostToDevice);
        tmp.destroy();
    }

    float *data;
    static constexpr int64_t row = Row;
    static constexpr int64_t col = Col;
    static constexpr int64_t lda = LDA;
    static constexpr bool row_major = RowMajor;
};

template<int64_t num_row, int64_t num_col>
__device__ StaticMatrix<num_row, CEILING(num_col, 4)> allocate_static() {
    return StaticMatrix<num_row, CEILING(num_col, 4)>();
}

template<int64_t len>
__device__ StaticArray<len> allocate_static_array() {
    return StaticArray<len>();
}

}  // namespace impl
