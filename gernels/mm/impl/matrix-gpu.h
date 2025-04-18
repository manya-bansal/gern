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

template<int Height, int Width, int LDA>
struct Placeholder {
    float *data;

    static constexpr int row = Height;
    static constexpr int col = Width;
    static constexpr int lda = LDA;

    __device__ __host__ float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
    }

    __device__ __host__ float &operator()(int64_t x, int64_t y) const {
        return data[x * lda + y];
    }

    __device__ void free_smem() {
        // sh_free(data);
    }

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
    static constexpr int row = Height;
    static constexpr int col = Width;

    template<int64_t q_height, int64_t q_width>
    __device__ Placeholder<q_height, q_width, width> get_view(int64_t x, int64_t y) {
        Placeholder<q_height, q_width, width> placeholder;
        placeholder.data = array + (x * width + y);
        return placeholder;
    }

    __device__ __host__ inline float &operator()(int64_t x, int64_t y) {
        return array[x * width + y];
    }

    __device__ __host__ inline float operator()(int64_t x, int64_t y) const {
        return array[x * width + y];
    }

    __device__ void free_smem() {
        // sh_free(data);
    }
};

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif
constexpr int URF{UNROLL_FACTOR};

#define CEILING(x, y) (((x) + (y) - 1) / (y))

namespace impl {

struct SharedMemoryManager {
    static constexpr int32_t max_size = 32 * 32 * 8 * 10;
    int32_t current_size = 0;

    __device__ void *malloc(int32_t size) {
        extern __shared__ char shmem[];
        void *to_return = shmem + current_size;
        current_size += size;
        return to_return;
    }

    __device__ void free() {
        current_size = 0;
    }
};

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
            data[i] = (float)(i % 10);
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

    template<int64_t num_row, int64_t num_col>
    inline __device__ StaticMatrixNoVector<num_row, num_col> query_2_reg_no_vector(int64_t x, int64_t y) {
        StaticMatrixNoVector<num_row, num_col> matrix;
        for (int64_t i = 0; i < num_row; i++) {
            for (int64_t j = 0; j < num_col; j++) {
                matrix(i, j) = operator()(x + i, y + j);
            }
        }
        return matrix;
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

    __host__ __device__ MatrixGPU(int32_t offset)
        : offset(offset) {
        CUDA_CHECK(cudaMalloc(&data, lda * row * sizeof(float)));
    }
    __host__ __device__ MatrixGPU(float *data, int32_t offset)
        : data(data), offset(offset) {
    }

    template<int64_t num_row, int64_t num_col>
    __device__ inline MatrixGPU<num_row, num_col, LDA, stride> query_global_2_global(int64_t x, int64_t y) {
        return MatrixGPU<num_row, num_col, LDA, stride>(data + (x * lda + y), offset);
    }

    template<int64_t num_row, int64_t num_col>
    __device__ inline MatrixGPU<num_row, num_col, LDA, stride> query_global_2_global_sync(int64_t x, int64_t y) {
        __syncthreads();
        return query_global_2_global<num_row, num_col>(x, y);
    }

    template<int64_t num_row, int64_t num_col>
    inline __device__ void insert_global_2_global(int64_t x, int64_t y, MatrixGPU<num_row, num_col, LDA, stride> to_insert) {
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
    __device__ inline StaticMatrixNoVector<num_row, num_col> query_2_reg_no_vector(int64_t x, int64_t y) {
        StaticMatrixNoVector<num_row, num_col> matrix;
        for (int64_t i = 0; i < num_row; i++) {
            for (int64_t j = 0; j < num_col; j++) {
                matrix(i, j) = data[(x + i) * lda + (y + j)];
            }
        }
        return matrix;
    }

    template<int64_t num_row, int64_t num_col>
    inline __device__ StaticMatrixNoVector<num_row, num_col> query_2_reg_no_vector_zero(int64_t x, int64_t y) {
        StaticMatrixNoVector<num_row, num_col> matrix;
        for (int64_t i = 0; i < num_row; i++) {
            for (int64_t j = 0; j < num_col; j++) {
                matrix(i, j) = 0.0f;
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
    __device__ void insert_from_reg_no_vector(int64_t x, int64_t y, StaticMatrixNoVector<num_row, num_col> to_insert) {
        auto temp = query_global_2_global<num_row, num_col>(x, y);
        for (uint resIdxM = 0; resIdxM < num_row; resIdxM += 1) {
            for (uint resIdxN = 0; resIdxN < num_col; resIdxN += 4) {
                // load C vector into registers
                float4 tmp = reinterpret_cast<float4 *>(
                    &temp(resIdxM, resIdxN))[0];

                // perform GEMM update in reg
                tmp.x = to_insert(resIdxM, resIdxN);
                tmp.y = to_insert(resIdxM, resIdxN + 1);
                tmp.z = to_insert(resIdxM, resIdxN + 2);
                tmp.w = to_insert(resIdxM, resIdxN + 3);
                // write back
                reinterpret_cast<float4 *>(
                    &temp(resIdxM, resIdxN))[0] =
                    tmp;
            }
        }
    }

    template<int64_t M, int64_t N, int64_t BLOCKSIZE>
    __device__ inline MatrixGPU<M, N, N, stride> query_global_2_shared(int64_t x,
                                                                       int64_t y,
                                                                       float *shmem) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N;

        auto a_q = query_global_2_global<M, N>(x, y);

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        if constexpr (total_elems == BLOCKSIZE) {
            int row = thread_id / N;
            int col = thread_id % N;
            shmem[thread_id] = operator()(x + row, y + col);
        } else {
            int innerRowA = thread_id / N;
            int innerColA = thread_id % N;
            constexpr int strideA = BLOCKSIZE / N;

            for (uint loadOffset = 0; loadOffset < M; loadOffset += strideA) {
                shmem[thread_id + loadOffset * N] =
                    a_q.data[(innerRowA + loadOffset) * a_q.lda + innerColA];
            }
        }
        return MatrixGPU<M, N, N, stride>(shmem, offset);
    }

    template<int64_t M, int64_t N, int64_t BLOCKSIZE>
    __device__ inline MatrixGPU<M, N, N, stride> query_global_2_shared(int64_t x,
                                                                       int64_t y) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N / (BLOCKSIZE);

        extern __shared__ char shmem[];
        float *shmem_malloced = (float *)(shmem);
        // current_size += total_elems * sizeof(float);

        auto a_q = query_global_2_global<M, N>(x, y);

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        if constexpr (total_elems == BLOCKSIZE) {
            int row = thread_id / N;
            int col = thread_id % N;
            shmem_malloced[thread_id] = operator()(x + row, y + col);
        } else {

            // for (int row_offset = innerRow; row_offset < total_elems / N; row_offset += stride) {
            //     for (int col = 0; col < N; ++col) {
            //         int idx = row_offset * N + col;
            //         shmem_malloced[idx] = a_q(row_offset, col);
            //     }
            // }
            for (int idx = thread_id; idx < total_elems; idx++) {

                int row = (idx / N);
                int col = idx % N;

                shmem_malloced[idx] = operator()(x + row, y + col);
            }
            // int innerRowA = thread_id / N;
            // int innerColA = thread_id % N;
            // constexpr int strideA = BLOCKSIZE / N;

            // for (uint loadOffset = 0; loadOffset < M; loadOffset += strideA) {
            //     shmem_malloced[thread_id + loadOffset * N] =
            //         a_q.data[(innerRowA + loadOffset) * a_q.lda + innerColA];
            // }
        }
        return MatrixGPU<M, N, N, stride>(shmem_malloced, offset);
    }

    template<int64_t M, int64_t N, int64_t BLOCKSIZE>
    __device__ inline MatrixGPU<N, M, M, 1> query_global_2_shared_vec_t(int64_t x,
                                                                        int64_t y,
                                                                        float *shmem) {

        int thread_id = threadIdx.x;
        constexpr int BLOCKSIZE_VEC = 4;

        static_assert(M % BLOCKSIZE_VEC == 0, "M must be divisible by BLOCKSIZE_VEC");

        constexpr int total_elems = (M * N) / BLOCKSIZE_VEC;

        constexpr int NUM_THREADS_VEC = BLOCKSIZE / BLOCKSIZE_VEC;

        const uint innerRowB = threadIdx.x / (N / BLOCKSIZE_VEC);
        const uint innerColB = threadIdx.x % (N / BLOCKSIZE_VEC);

        // static_assert(BLOCKSIZE == total_elems, "BLOCKSIZE must be equal to total_elems");

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        if constexpr (BLOCKSIZE == total_elems) {
            int row = thread_id / N;
            int col = thread_id % N;
            float4 tmp =
                reinterpret_cast<float4 *>(&operator()(x + innerRowB, y + innerColB * 4))[0];
            shmem[(innerColB * 4 + 0) * M + innerRowB] = tmp.x;
            shmem[(innerColB * 4 + 1) * M + innerRowB] = tmp.y;
            shmem[(innerColB * 4 + 2) * M + innerRowB] = tmp.z;
            shmem[(innerColB * 4 + 3) * M + innerRowB] = tmp.w;

        } else {
            constexpr int strideB = (BLOCKSIZE * 4) / N;
            auto temp = query_global_2_global<M, N>(x, y);

            for (uint offset = 0; offset + strideB <= M; offset += strideB) {
                const float4 tmp = reinterpret_cast<const float4 *>(
                    &temp(innerRowB + offset, innerColB * 4))[0];
                // float4 tmp;
                // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
                //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
                //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
                shmem[(innerColB * 4 + 0) * M + innerRowB + offset] = tmp.x;
                shmem[(innerColB * 4 + 1) * M + innerRowB + offset] = tmp.y;
                shmem[(innerColB * 4 + 2) * M + innerRowB + offset] = tmp.z;
                shmem[(innerColB * 4 + 3) * M + innerRowB + offset] = tmp.w;
            }
        }

        return MatrixGPU<N, M, M, 1>(shmem);
    }

    template<int64_t M, int64_t N>
    __device__ inline MatrixGPU<M, N, N, 1> query_global_2_shared_restrict(int64_t x,
                                                                           int64_t y) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N;

        extern __shared__ char shmem[];
        float *shmem_malloced = (float *)(shmem);
        // current_size += total_elems * sizeof(float);

        int row = thread_id / N;
        int col = thread_id % N;
        shmem_malloced[thread_id] = operator()(x + row, y + col);
        return MatrixGPU<M, N, N, 1>(shmem_malloced, offset);
    }

    template<int64_t M, int64_t N>
    __device__ inline MatrixGPU<M, N, N, 1> query_global_2_shared_no_temp(int64_t x,
                                                                          int64_t y) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N;

        extern __shared__ char shmem[];
        float *shmem_malloced = (float *)(shmem + offset);
        int block_dim = blockDim.x;

        // Idk why this constexpr is needed, and why the compiler does not optimize this
#pragma unroll 8
        for (int idx = thread_id; idx < total_elems; idx += block_dim) {
            int row = idx / N;
            int col = idx % N;

            shmem_malloced[idx] = operator()(x + row, y + col);
        }

        return MatrixGPU<M, N, N, 1>(shmem_malloced);
    }

    template<int64_t num_row, int64_t num_col>
    __device__ MatrixGPUShared<num_row, num_col, num_col> stage_into_smem(int64_t x, int64_t y) {
        __shared__ float *smem_data_global;

        __syncthreads();

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
    __device__ MatrixGPUShared<num_row, num_col, num_col> stage_into_smem_flat(int64_t x, int64_t y) {
        __shared__ float *smem_data_global;

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            smem_data_global = (float *)sh_malloc(num_row * num_col * sizeof(float));
        }
        __syncthreads();

        int thread_id = threadIdx.x;
        int total_elems = num_row * num_col;

        for (int idx = thread_id; idx < total_elems; idx += blockDim.x) {
            int row = idx / num_col;
            int col = idx % num_col;

            float *start = smem_data_global;
            float *data_start = data + (x + row) * lda + (y + col);
            start[idx] = data_start[0];
        }

        __syncthreads();

        return MatrixGPUShared<num_row, num_col, num_col>(smem_data_global);
    }

    template<int64_t num_row, int64_t num_col>
    __device__ MatrixGPUShared<num_row, num_col, num_col> stage_into_smem_vec(int64_t x, int64_t y) {
        static_assert(num_col % 4 == 0, "num_col must be divisible by 4");
        static_assert(lda % 4 == 0, "num_col must be divisible by 4");
        __shared__ float *smem_data_global;

        __syncthreads();

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
    inline __device__ void insert(int x,
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

    inline __device__ float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
    }

    inline __device__ float operator()(int64_t x, int64_t y) const {
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
    int32_t offset = 0;
    // impl::SharedMemoryManager &smem_manager;
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