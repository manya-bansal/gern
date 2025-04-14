#pragma once

#include "matrix-gpu.h"

// For reference.
void matrix_multiply_cpu(const impl::MatrixCPU &A,
                         const impl::MatrixCPU &B,
                         impl::MatrixCPU &C) {
    for (int64_t i = 0; i < A.row; i++) {
        for (int64_t j = 0; j < B.col; j++) {
            for (int64_t k = 0; k < A.col; k++) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

template<int64_t k_dim, typename AT, typename BT, typename CT>
inline __device__ void matrix_multiply_reg(const AT &A,
                                           const BT &B,
                                           CT &C) {

    const float *b_flat = (const float *)B.array;
    constexpr int64_t B_cols = B.cols_by_4 * 4;

    for (int64_t i = 0; i < C.rows; i++) {
        for (int64_t j = 0; j < C.cols_by_4; j++) {
            float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int64_t k = 0; k < A.cols_by_4; k++) {
                float4 a = A.array[i * A.cols_by_4 + k];
                for (int64_t l = 0; l < 4; l++) {
                    int64_t k4 = k * 4;
                    int64_t j4 = j * 4;
                    tmp[l] += a.x * b_flat[k4 * B_cols + j4 + l];
                    tmp[l] += a.y * b_flat[(k4 + 1) * B_cols + j4 + l];
                    tmp[l] += a.z * b_flat[(k4 + 2) * B_cols + j4 + l];
                    tmp[l] += a.w * b_flat[(k4 + 3) * B_cols + j4 + l];
                }
            }

            float4 res = *(float4 *)tmp;
            C.array[i * C.cols_by_4 + j].x += res.x;
            C.array[i * C.cols_by_4 + j].y += res.y;
            C.array[i * C.cols_by_4 + j].z += res.z;
            C.array[i * C.cols_by_4 + j].w += res.w;
        }
    }
}

template<int64_t k_dim, typename AT, typename BT, typename CT>
inline __device__ void matrix_multiply(const AT &A,
                                       const BT &B,
                                       CT &C) {
    float tmp = 0.0f;
    for (int64_t i = 0; i < A.row; i++) {
        for (int64_t j = 0; j < B.col; j++) {
            for (int64_t k = 0; k < k_dim; k++) {
                tmp += A(i, k) * B(k, j);
            }
            C(i, j) += tmp;
        }
    }
}

template<int64_t k_dim, typename AT, typename BT, typename CT>
inline __device__ void matrix_multiply_warp(const AT &A,
                                            const BT &B,
                                            CT &C) {
    if (threadIdx.x % 32 == 0 && threadIdx.y % 32 == 0) {
        matrix_multiply<k_dim>(A, B, C);
    }
}
