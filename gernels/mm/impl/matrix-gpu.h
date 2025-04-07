#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#include "sh_malloc.h"

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
    // template<int num_row, int num_col>
    // __device__ StaticMatrix<num_row, num_col / 4> query_new(int64_t x, int64_t y) {
    //     return *this;
    // }
};

template<int Height, int Width>
struct StaticMatrixNoVector {
    float array[Height * Width];
    static constexpr int height = Height;
    static constexpr int width = Width;

    __device__ __host__ float &operator()(int64_t x, int64_t y) {
        return array[x * width + y];
    }

    __device__ __host__ float operator()(int64_t x, int64_t y) const {
        return array[x * width + y];
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

    float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
    }

    float operator()(int64_t x, int64_t y) const {
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
            data[i] = (float)(i % 100);
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

template<int64_t Height, int64_t Width, int LDA>
class MatrixGPUShared {
public:
    __device__ MatrixGPUShared(float *data)
        : data(data) {
    }

    __device__ float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
    }

    __device__ float operator()(int64_t x, int64_t y) const {
        return data[x * lda + y];
    }

    __device__ void free_smem() {
        // sh_free(data);
    }
    template<int64_t q_height, int64_t q_width>
    __device__ MatrixGPUShared<q_height, q_width, LDA> get_view(int64_t x, int64_t y) {

        return MatrixGPUShared<q_height, q_width, LDA>(data + (x * lda + y));
    }

    template<int64_t q_height, int64_t q_width>
    __device__ MatrixGPUShared<q_height, q_width, LDA> get_view_vec(int64_t x, int64_t y) {

        return MatrixGPUShared<q_height, q_width, LDA>(data + (x * lda + y));
    }

    float *data;
    static constexpr int64_t row = Height;
    static constexpr int64_t col = Width;
    static constexpr int64_t lda = LDA;
};

template<int64_t Row, int64_t Col, int64_t LDA, int64_t stride>
class MatrixGPU {
public:
    MatrixGPU() {
        CUDA_CHECK(cudaMalloc(&data, lda * row * sizeof(float)));
    }

    __device__ MatrixGPU(float *data)
        : data(data) {
    }

    template<int64_t num_row, int64_t num_col>
    __device__ MatrixGPU<num_row, num_col, LDA, stride> query_global_2_global(int64_t x, int64_t y) {
        return MatrixGPU<num_row, num_col, LDA, stride>(data + (x * lda + y));
    }

    template<int64_t num_row, int64_t num_col>
    __device__ void insert_global_2_global(int64_t x, int64_t y, MatrixGPU<num_row, num_col, LDA, stride> to_insert) {
        float *data_tmp;
        float *data_insert_tmp;
        for (int64_t i = 0; i < num_row; i++) {
            data_tmp = data + ((x + i) * lda) + y;
            data_insert_tmp = to_insert.data + (i * to_insert.lda);
            for (int64_t j = 0; j < num_col; j++) {
                data_tmp[j] = data_insert_tmp[j];
            }
        }
    }

    template<int64_t num_row, int64_t num_col>
    __device__ StaticMatrixNoVector<num_row, num_col> query_2_reg_no_vector(int64_t x, int64_t y) {
        StaticMatrixNoVector<num_row, num_col> matrix;
        for (int64_t i = 0; i < num_row; i++) {
            for (int64_t j = 0; j < num_col; j++) {
                matrix(i, j) = data[(x + i) * lda + (y + j)];
            }
        }
        return matrix;
    }

    template<int64_t num_row, int64_t num_col>
    __device__ void insert_2_reg_no_vector(int64_t x, int64_t y, StaticMatrixNoVector<num_row, num_col> to_insert) {
        for (int64_t i = 0; i < num_row; i++) {
            for (int64_t j = 0; j < num_col; j++) {
                data[(x + i) * lda + (y + j)] = to_insert(i, j);
            }
        }
    }

    template<int64_t num_row, int64_t num_col>
    __device__ MatrixGPUShared<num_row, num_col, num_col> stage_into_smem(int64_t x, int64_t y) {
        __shared__ float *smem_data_global;

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            smem_data_global = (float *)sh_malloc(num_row * num_col * sizeof(float));
        }
        __syncthreads();

        for (int64_t i = 0; i < num_row; i += blockDim.x) {
            for (int64_t j = 0; j < num_col; j += blockDim.y) {
                float *start = smem_data_global + (i * num_col + j);
                float *data_start = data + (x + i + threadIdx.x) * lda + (y + j + threadIdx.y);
                start[(threadIdx.x * num_col) + threadIdx.y] = data_start[0];
            }
        }

        __syncthreads();

        return MatrixGPUShared<num_row, num_col, num_col>(smem_data_global);
    }

    template<int64_t num_row, int64_t num_col>
    __device__ MatrixGPUShared<num_row, num_col, num_col> stage_into_smem_vec(int64_t x, int64_t y) {
        static_assert(num_col % 4 == 0, "num_col must be divisible by 4");
        static_assert(lda % 4 == 0, "num_col must be divisible by 4");
        __shared__ float *smem_data_global;

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            smem_data_global = (float *)sh_malloc(num_row * num_col * sizeof(float));
        }
        __syncthreads();

        // Get to the start of the data.
        float4 *val_ptr = reinterpret_cast<float4 *>(data + (x * lda + y));
        float4 *smem_val_ptr = reinterpret_cast<float4 *>(smem_data_global);

        for (int64_t i = 0; i < num_row; i += blockDim.x) {
            for (int64_t j = 0; j < num_col / 4; j += blockDim.y) {
                smem_val_ptr[threadIdx.x * (num_col / 4) + threadIdx.y] = val_ptr[threadIdx.x * (lda / 4) + threadIdx.y];
            }
            val_ptr += blockDim.x * lda / 4;
            smem_val_ptr += blockDim.x * num_col / 4;
        }

        __syncthreads();

        return MatrixGPUShared<num_row, num_col, num_col>(smem_data_global);
    }

    template<int64_t num_row, int64_t num_col, int q_lda>
    __device__ void insert_from_smem(int64_t x, int64_t y, MatrixGPUShared<num_row, num_col, q_lda> m) {

        float *data_start = data + x * lda + y;

        for (int64_t i = 0; i < num_row; i += blockDim.x) {
            for (int64_t j = 0; j < num_col; j += blockDim.y) {
                float *m_start = m.data + (i + threadIdx.x) * m.lda + (j + threadIdx.y);
                data_start[threadIdx.x * lda + threadIdx.y] = m_start[0];
            }
            data_start += lda * blockDim.x;
        }

        __syncthreads();
    }

    __device__ void free_smem() {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            sh_free(data);
        }
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

    __device__ float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
    }

    __device__ float operator()(int64_t x, int64_t y) const {
        return data[x * lda + y];
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
        CUDA_CHECK(cudaMemcpy(cpu.data, data, lda * row * sizeof(float), cudaMemcpyDeviceToHost));
        return cpu;
    }

    void destroy() {
        CUDA_CHECK(cudaFree(data));
    }

    void vvals(float f) {
        MatrixCPU tmp(row, col, lda);
        tmp.vvals(f);
        CUDA_CHECK(cudaMemcpy(data, tmp.data, lda * row * sizeof(float), cudaMemcpyHostToDevice));
        tmp.destroy();
    }

    void ascending() {
        MatrixCPU tmp(row, col, lda);
        tmp.ascending();
        CUDA_CHECK(cudaMemcpy(data, tmp.data, lda * row * sizeof(float), cudaMemcpyHostToDevice));
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