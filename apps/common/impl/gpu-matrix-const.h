#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define CUDA_CHECK(x)                                                                    \
    {                                                                                    \
        cudaError_t e = x;                                                               \
        if (e != cudaSuccess) {                                                          \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

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

#define CEILING(x, y) (((x) + (y)-1) / (y))

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

    float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
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
            data[i] = i;
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
    os << "["
       << "\n";
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

template<int64_t Row, int64_t Col, int64_t LDA, int64_t stride>
class MatrixGPU {
public:
    MatrixGPU() {
        CUDA_CHECK(cudaMalloc(&data, lda * row * sizeof(float)));
    }

    template<int64_t num_row, int64_t num_col>
    __device__ StaticMatrix<num_row, CEILING(num_col, 4)> query(int64_t x, int64_t y) {
        constexpr int64_t col_to_return = CEILING(num_col, 4);
        StaticMatrix<num_row, col_to_return> matrix;
        auto matrix_data = matrix.array;
        float *data_ptr = &data[x * row + y];
        float4 *val_ptr = reinterpret_cast<float4 *>(data_ptr);

        for (int m = 0; m < num_row; m++) {
#pragma unroll URF
            for (int i = 0; i < col_to_return; i += stride) {
                float4 val = val_ptr[i];
                matrix_data[i] = val;
            }
            val_ptr += LDA / 4;

            matrix_data += col_to_return;
        }
        return matrix;
    }

    template<typename T2>
    __device__ void insert(int x,
                           int y,
                           T2 &reg_array_big) {
        auto matrix_data = reg_array_big.array;
        constexpr int64_t num_row = reg_array_big.rows;
        constexpr int64_t num_col = reg_array_big.cols_by_4;
        float4 *val_ptr = reinterpret_cast<float4 *>(&data[x * row + y]);

        for (int m = 0; m < num_row; m++) {
#pragma unroll URF
            for (int i = 0; i < num_col; i += stride) {
                float4 val = matrix_data[i];
                val_ptr[i] = val;
            }
            val_ptr += (LDA / 4);
            matrix_data += num_col;
        }
    }

    template<
        typename T2>
    __device__ void insert_new(
        int x,
        int y,
        T2 &reg_array_big) {
        y = y / (col / stride);
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

    template<int64_t num_row_t, int64_t num_col_t>
    __device__ StaticMatrix<num_row_t, CEILING(num_col_t, 4)> query_new(
        int x,
        int y) {
        y = y / (col / stride);
        constexpr int64_t col_to_return = CEILING(num_col_t, 4);
        StaticMatrix<num_row_t, col_to_return> reg_array_big;

        auto &reg_array = reg_array_big.array;
        float4 *val_ptr = reinterpret_cast<float4 *>(&data[x * row + y * 4]);

        constexpr int64_t num_row = reg_array_big.rows;
        constexpr int64_t num_col = reg_array_big.cols_by_4;

        int index = 0;
        for (int m = 0; m < num_row; m++) {
#pragma unroll URF
            for (int i = 0; i < num_col; i++) {
                float4 val = val_ptr[index];
                reg_array[i] = val;
                index += stride;
            }
        }
        return reg_array_big;
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
