#pragma once

#include "cpu-matrix.h"
#include "library/smem_allocator/sh_malloc.h"

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

    __device__ MatrixGPU stage_into_smem(int64_t x, int64_t y, int64_t l_x, int64_t l_y) {
        __shared__ float *smem_data_global;

        if (threadIdx.x == 0) {
            smem_data_global = (float *)sh_malloc(l_x * l_y * sizeof(float));
        }
        __syncthreads();

        // float *data_start = data + (x)*lda + (y);
        for (int64_t i = 0; i < l_x; i += blockDim.x) {
            for (int64_t j = 0; j < l_y; j += blockDim.y) {
                float *start = smem_data_global + (i * l_y + j);
                float *data_start = data + (x + i + threadIdx.x) * lda + (y + j + threadIdx.y);
                start[(threadIdx.x * l_y) + threadIdx.y] = data_start[0];
            }
            // data_start += lda * blockDim.x;
        }

        __syncthreads();
        return MatrixGPU(smem_data_global, l_x, l_y, l_y);
    }

    __device__ void insert(int64_t x, int64_t y, int64_t l_x, int64_t l_y, MatrixGPU m) {

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("Inserting matrix at %lld, %lld\n", x, y);
            for (int64_t i = 0; i < l_x; i++) {
                for (int64_t j = 0; j < l_y; j++) {
                    printf("%f \n", m.data[i * l_y + j]);
                }
            }
        }

        for (int64_t i = 0; i < l_x; i += blockDim.x) {
            for (int64_t j = 0; j < l_y; j += blockDim.y) {
                float *m_start = m.data + (i + threadIdx.x) * m.lda + (j + threadIdx.y);
                float *data_start = data + (x + i + threadIdx.x) * lda + (y + j + threadIdx.y);
                data_start[0] = m_start[0];
            }
        }

        __syncthreads();
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
        MatrixCPU tmp(row, col, lda);
        tmp.ascending();
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

__device__ inline void add_smem(MatrixGPU a, MatrixGPU b) {
    float *a_data;
    float *b_data;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Inserting matrix at %lld, %lld\n", a.row, a.col);
        for (int64_t i = 0; i < a.row; i += 2) {
            for (int64_t ii = 0; ii < 2; ii++) {
                for (int64_t j = 0; j < a.col; j++) {
                    printf("%f \n", a.data[(i + ii) * a.lda + j]);
                }
            }
        }
    }

    for (int64_t i = 0; i < a.row; i += blockDim.x) {
        for (int64_t j = 0; j < a.col; j += blockDim.y) {
            a_data = a.data + (i + threadIdx.x) * a.lda + (j + threadIdx.y);
            b_data = b.data + (i + threadIdx.x) * b.lda + (j + threadIdx.y);
            b_data[0] = a_data[0] + 1;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Inserting matrix at %lld, %lld\n", a.row, a.col);
        printf("blockDim.x: %u, blockDim.y: %u\n", blockDim.x, blockDim.y);
        for (int64_t i = 0; i < a.row; i++) {
            for (int64_t j = 0; j < a.col; j++) {
                printf("%f \n", b.data[i * b.lda + j]);
            }
        }
    }
    __syncthreads();
}

}  // namespace impl
}  // namespace gern
