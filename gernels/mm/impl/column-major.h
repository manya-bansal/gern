#pragma once

#include "matrix-gpu.h"
#include <cuda_runtime.h>

namespace impl {

template<int64_t NumRow, int64_t NumCol, int64_t Lda = NumRow>
class ColumnMajorMatrix {
public:
    ColumnMajorMatrix() {
        cudaMalloc(&data, sizeof(float) * NumRow * NumCol);
    }
    __device__ ColumnMajorMatrix(float *data)
        : data(data) {
    }

    __host__ __device__ ColumnMajorMatrix(int32_t offset)
        : data(data), offset(offset) {
        cudaMalloc(&data, sizeof(float) * NumRow * NumCol);
    }

    __host__ __device__ ColumnMajorMatrix(float *data, int32_t offset)
        : data(data), offset(offset) {
    }

    void ascending() {
        float *data_tmp = (float *)malloc(sizeof(float) * NumRow * NumCol);
        for (int64_t i = 0; i < NumRow; i++) {
            for (int64_t j = 0; j < NumCol; j++) {
                data_tmp[i * lda + j] = i % 10;
            }
        }
        cudaMemcpy(data, data_tmp, sizeof(float) * NumRow * NumCol, cudaMemcpyHostToDevice);
        free(data_tmp);
    }

    void vvals(float f) {
        float *data_tmp = (float *)malloc(sizeof(float) * NumRow * NumCol);
        for (int64_t i = 0; i < NumRow; i++) {
            for (int64_t j = 0; j < NumCol; j++) {
                data_tmp[i * lda + j] = f;
            }
        }
        cudaMemcpy(data, data_tmp, sizeof(float) * NumRow * NumCol, cudaMemcpyHostToDevice);
        free(data_tmp);
    }

    inline __device__ float &operator()(int64_t x, int64_t y) {
        return data[x * lda + y];
    }

    inline __device__ float operator()(int64_t x, int64_t y) const {
        return data[x * lda + y];
    }

    template<int64_t M, int64_t N>
    __device__ inline ColumnMajorMatrix<M, N, Lda> query_global_2_global(int64_t x, int64_t y) {
        return ColumnMajorMatrix<M, N, Lda>(&operator()(x, y), offset);
    }

    template<int64_t M, int64_t N, int64_t BLOCKSIZE>
    __device__ inline MatrixGPU<M, N, N, 1> query_global_2_shared(int64_t x,
                                                                  int64_t y,
                                                                  float *shmem) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N;

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        if constexpr (total_elems == BLOCKSIZE) {
            int row = thread_id / N;
            int col = thread_id % N;
            shmem[thread_id] = operator()(x + row, y + col);
        } else {
            // #pragma unroll 8
            //             for (int idx = thread_id; idx < total_elems; idx += BLOCKSIZE) {
            //                 int row = idx / N;
            //                 int col = idx % N;

            //                 shmem[idx] = operator()(x + row, y + col);
            //             }
            auto temp = query_global_2_global<M, N>(x, y);
            int innerRowB = thread_id / N;
            int innerColB = thread_id % N;
            constexpr int strideB = BLOCKSIZE / N;

            for (uint loadOffset = 0; loadOffset < M; loadOffset += strideB) {
                shmem[(innerRowB + loadOffset) * N + innerColB] =
                    // temp.data[(innerRowB + loadOffset) * N + innerColB];
                    temp(innerRowB + loadOffset, innerColB);
                // temp(innerRowA + loadOffset, innerColA);
            }
        }

        return MatrixGPU<M, N, N, 1>(shmem);
    }

    template<int64_t M, int64_t N, int64_t BLOCKSIZE>
    __device__ inline MatrixGPU<M, N, N, 1> query_global_2_shared_vec(int64_t x,
                                                                      int64_t y,
                                                                      float *shmem) {

        int thread_id = threadIdx.x;
        constexpr int BLOCKSIZE_VEC = 4;

        static_assert(M % BLOCKSIZE_VEC == 0, "M must be divisible by BLOCKSIZE_VEC");

        constexpr int total_elems = (M * N) / BLOCKSIZE_VEC;

        constexpr int NUM_THREADS_VEC = BLOCKSIZE / BLOCKSIZE_VEC;
        constexpr int strideB = NUM_THREADS_VEC / N;

        const uint innerRowB = threadIdx.x / (N / BLOCKSIZE_VEC);
        const uint innerColB = threadIdx.x % (N / BLOCKSIZE_VEC);

        // static_assert(BLOCKSIZE == total_elems, "NUM_THREADS_VEC must be equal to total_elems");

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        if constexpr (BLOCKSIZE == total_elems) {
            int row = thread_id / N;
            int col = thread_id % N;
            // shmem[thread_id] = operator()(x + row, y + col);
            reinterpret_cast<float4 *>(&shmem[innerRowB * N + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(data + (x + innerRowB) * lda + (y + innerColB * 4))[0];
        } else {
            constexpr int strideB = (BLOCKSIZE * 4) / N;
            auto temp = query_global_2_global<M, N>(x, y);

            for (uint loadOffset = 0; loadOffset + strideB <= M; loadOffset += strideB) {
                reinterpret_cast<float4 *>(
                    &shmem[(innerRowB + loadOffset) * N + innerColB * 4])[0] =
                    reinterpret_cast<const float4 *>(
                        &temp(innerRowB + loadOffset, innerColB * 4))[0];
            }
        }

        return MatrixGPU<M, N, N, 1>(shmem);
    }

    template<int64_t M, int64_t N, int64_t BLOCKSIZE>
    __device__ inline MatrixGPU<M, N, N, BLOCKSIZE> query_global_2_shared(int64_t x,
                                                                          int64_t y) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N;

        extern __shared__ char shmem[];
        float *shmem_malloced = (float *)(shmem + offset);
        // current_size += total_elems * sizeof(float);

        // MatrixGPUSharedStatic<M, N, N> ret;
        // float *shmem_malloced = ret.data;

        // float *shmem_malloced = (float *)smem_manager->malloc(total_elems * sizeof(float));
        // float *Bs = (float *)smem_manager.malloc(BLOCKSIZE * BLOCKSIZE * sizeof(float));

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        if constexpr (total_elems == BLOCKSIZE) {
            int row = thread_id / N;
            int col = thread_id % N;
            shmem_malloced[thread_id] = operator()(x + row, y + col);
        } else {
#pragma unroll 8
            for (int idx = thread_id; idx < total_elems; idx += BLOCKSIZE) {
                int row = idx / N;
                int col = idx % N;

                shmem_malloced[idx] = operator()(x + row, y + col);
            }
        }

        return MatrixGPU<M, N, N, BLOCKSIZE>(shmem_malloced);
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

    template<int64_t M, int64_t N>
    __device__ inline MatrixGPU<M, N, N, 1> query_global_2_shared_restrict(int64_t x,
                                                                           int64_t y) {
        int thread_id = threadIdx.x;
        constexpr int total_elems = M * N;

        extern __shared__ char shmem[];
        float *shmem_malloced = (float *)(shmem + offset);
        // current_size += total_elems * sizeof(float);

        // MatrixGPUSharedStatic<M, N, N> ret;
        // float *shmem_malloced = ret.data;

        // float *shmem_malloced = (float *)smem_manager->malloc(total_elems * sizeof(float));
        // float *Bs = (float *)smem_manager.malloc(BLOCKSIZE * BLOCKSIZE * sizeof(float));

        // Idk why this constexpr is needed, and why the compiler does not optimize this
        int row = thread_id / N;
        int col = thread_id % N;
        shmem_malloced[thread_id] = operator()(x + row, y + col);

        return MatrixGPU<M, N, N, 1>(shmem_malloced);
    }

    MatrixCPU get() {
        MatrixCPU cpu(NumRow, NumCol, NumCol);
        cudaMemcpy(cpu.data, data, sizeof(float) * NumCol * lda, cudaMemcpyDeviceToHost);
        return cpu;
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
    __device__ void insert_2_reg_no_vector(int64_t x, int64_t y, const StaticMatrixNoVector<num_row, num_col> &to_insert) {
        for (int64_t i = 0; i < num_row; ++i) {
            for (int64_t j = 0; j < num_col; ++j) {
                data[(x + i) * lda + (y + j)] = to_insert(i, j);
            }
        }
    }

    template<int64_t num_row, int64_t num_col>
    __device__ inline void insert_2_reg_vector(int64_t x, int64_t y, const StaticMatrixNoVector<num_row, num_col> &to_insert) {
        static_assert(num_col % 4 == 0, "Number of columns must be divisible by 4 for float4 operations.");

        auto temp = query_global_2_global<num_row, num_col>(x, y);

        for (int64_t i = 0; i < num_row; ++i) {
            for (int64_t j = 0; j < num_col; j += 4) {
                float4 tmp;
                tmp.x = to_insert(i, j);
                tmp.y = to_insert(i, j + 1);
                tmp.z = to_insert(i, j + 2);
                tmp.w = to_insert(i, j + 3);

                // Calculate the linear index for row-major storage
                reinterpret_cast<float4 *>(&temp(i, j))[0] = tmp;
            }
        }
    }

    void destroy() {
        cudaFree(data);
    }

    static constexpr int64_t num_row = NumRow;
    static constexpr int64_t num_col = NumCol;
    static constexpr int64_t lda = Lda;
    float *data;
    int32_t offset;
};

}  // namespace impl