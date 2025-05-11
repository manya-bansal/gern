#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdlib.h>

#include "kernel/kernel.h"

namespace gern {
namespace impl {

namespace internal {
constexpr int64_t matrix_alignment = 64;
constexpr int64_t matrix_alignment_in_floats = matrix_alignment / sizeof(float);
constexpr int64_t kernel_rows = 32;
constexpr int64_t kernel_cols = 12;

enum class Layout {
    RowMajor,
    ColumnMajor
};

class MatrixViewBase {
protected:
    MatrixViewBase(float *data, int64_t l_x, int64_t l_y, int64_t lda)
        : data(data), row(l_x), col(l_y), lda(lda) {
    }
    MatrixViewBase() = default;

public:
    float *data;
    int64_t row;
    int64_t col;
    int64_t lda;

    void destroy() {
        free(this->data);
    }
};

class MatrixViewRowMajor : public MatrixViewBase {
public:
    static constexpr Layout layout = Layout::RowMajor;
    MatrixViewRowMajor(float *data, int64_t l_x, int64_t l_y, int64_t lda)
        : MatrixViewBase(data, l_x, l_y, lda) {
    }
    MatrixViewRowMajor(int64_t l_x, int64_t l_y) {
        this->row = l_x;
        this->col = l_y;
        this->lda = (l_y + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
        this->data = (float *)aligned_alloc(matrix_alignment, l_x * this->lda * sizeof(float));
    }

    float &get(int64_t x, int64_t y) {
        return this->data[x * this->lda + y];
    }
};

class MatrixViewColumnMajor : public MatrixViewBase {
public:
    static constexpr Layout layout = Layout::ColumnMajor;
    MatrixViewColumnMajor(float *data, int64_t l_x, int64_t l_y, int64_t lda)
        : MatrixViewBase(data, l_x, l_y, lda) {
    }
    MatrixViewColumnMajor(int64_t l_x, int64_t l_y) {
        this->row = l_x;
        this->col = l_y;
        this->lda = (l_x + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
        this->data = (float *)aligned_alloc(matrix_alignment, l_y * this->lda * sizeof(float));
    }

    float &get(int64_t x, int64_t y) {
        return this->data[y * this->lda + x];
    }
};

template<typename Derived, typename ViewType>
class QueryView {
public:
    ViewType query(int64_t x, int64_t y, int64_t l_x, int64_t l_y) {
        if constexpr (Derived::layout != ViewType::layout) {
            static_assert(false, "QueryView can only be used with matching layouts");
        }
        Derived *self = static_cast<Derived *>(this);
        return ViewType(&self->get(x, y), l_x, l_y, self->lda);
    }
};

template<int64_t L_X, int64_t L_Y>
class MatrixStaticMemRowMajor {
protected:
    static constexpr int64_t matrix_alignment_in_floats = matrix_alignment / sizeof(float);

public:
    static constexpr Layout layout = Layout::RowMajor;
    static constexpr int64_t row = L_X;
    static constexpr int64_t col = L_Y;
    static constexpr int64_t lda = (L_Y + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
    alignas(matrix_alignment) float data[row * lda];

    MatrixStaticMemRowMajor() = default;
    float &get(int64_t x, int64_t y) {
        return this->data[x * this->lda + y];
    }
};

template<int64_t L_X, int64_t L_Y>
class MatrixStaticPackedForKernelLeft {
protected:
    static constexpr int64_t size_per_microkernel = kernel_rows * L_Y;
    static constexpr int64_t total_size = size_per_microkernel * (L_X / kernel_rows);

public:
    static constexpr Layout layout = Layout::ColumnMajor;
    static constexpr int64_t lda = kernel_rows;

    static constexpr int64_t row = L_X;
    static constexpr int64_t col = L_Y;

    alignas(matrix_alignment) float data[total_size];
    MatrixStaticPackedForKernelLeft() = default;

    float &get(int64_t x, int64_t y) {
        int64_t row_group = x / kernel_rows;
        int64_t row_idx = x % kernel_rows;

        int64_t group_begin = row_group * size_per_microkernel;
        int64_t idx_in_group = y * kernel_rows + row_idx;

        return this->data[group_begin + idx_in_group];
    }
};

template<int64_t L_X, int64_t L_Y>
class MatrixStaticPackedForKernelRight {
protected:
    static constexpr int64_t size_per_microkernel = kernel_cols * L_X;
    static constexpr int64_t total_size = size_per_microkernel * (L_Y / kernel_cols);

public:
    static constexpr Layout layout = Layout::RowMajor;
    static constexpr int64_t lda = kernel_cols;

    static constexpr int64_t row = L_X;
    static constexpr int64_t col = L_Y;

    alignas(matrix_alignment) float data[total_size];
    MatrixStaticPackedForKernelRight() = default;

    float &get(int64_t x, int64_t y) {
        int64_t col_group = y / kernel_cols;
        int64_t col_idx = y % kernel_cols;

        int64_t group_begin = col_group * size_per_microkernel;
        int64_t idx_in_group = x * kernel_cols + col_idx;

        return this->data[group_begin + idx_in_group];
    }
};

template<int64_t L_X, int64_t L_Y>
class MatrixStaticMemColumnMajor {
public:
    static constexpr Layout layout = Layout::ColumnMajor;
    static constexpr int64_t row = L_X;
    static constexpr int64_t col = L_Y;
    static constexpr int64_t lda = (L_X + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
    alignas(matrix_alignment) float data[col * lda];

    MatrixStaticMemColumnMajor() = default;
    float &get(int64_t x, int64_t y) {
        return this->data[y * this->lda + x];
    }
};

template<typename Derived, template<int64_t, int64_t> class StaticMemType>
class QueryStaticMem {
public:
    template<int64_t L_X, int64_t L_Y>
    StaticMemType<L_X, L_Y> query(int64_t x, int64_t y) {
        StaticMemType<L_X, L_Y> res;

        if constexpr (Derived::layout == Layout::RowMajor) {
            for (int i = 0; i < L_X; i++) {
                for (int j = 0; j < L_Y; j++) {
                    res.get(i, j) = static_cast<Derived *>(this)->get(x + i, y + j);
                }
            }
        } else {
            for (int j = 0; j < L_Y; j++) {
                for (int i = 0; i < L_X; i++) {
                    res.get(i, j) = static_cast<Derived *>(this)->get(x + i, y + j);
                }
            }
        }
        return res;
    }

    template<int64_t L_X, int64_t L_Y>
    void insert(int64_t x, int64_t y, StaticMemType<L_X, L_Y> &to_insert) {
        if constexpr (Derived::layout == Layout::RowMajor) {
            for (int i = 0; i < L_X; i++) {
                for (int j = 0; j < L_Y; j++) {
                    static_cast<Derived *>(this)->get(x + i, y + j) = to_insert.get(i, j);
                }
            }
        } else {
            for (int j = 0; j < L_Y; j++) {
                for (int i = 0; i < L_X; i++) {
                    static_cast<Derived *>(this)->get(x + i, y + j) = to_insert.get(i, j);
                }
            }
        }
    }
};

}  // namespace internal

// Here are the actual types that will be used
class MatrixViewBasic : public internal::MatrixViewRowMajor, public internal::QueryView<MatrixViewBasic, MatrixViewBasic> {
    MatrixViewBasic(int64_t l_x, int64_t l_y)
        : MatrixViewRowMajor(l_x, l_y) {
    }

public:
    MatrixViewBasic(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static MatrixViewBasic allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return MatrixViewBasic(l_x, l_y);
    }
};

template<int64_t L_X, int64_t L_Y>
class MatrixTemplated : public internal::MatrixStaticMemRowMajor<L_X, L_Y>, public internal::QueryView<MatrixTemplated<L_X, L_Y>, MatrixViewBasic> {
public:
    MatrixTemplated() = default;
    void destroy() {
    }
    static MatrixTemplated allocate(int64_t, int64_t, int64_t, int64_t) {
        return MatrixTemplated();
    }
};

class MatrixViewPackOnQuery : public internal::MatrixViewRowMajor, public internal::QueryStaticMem<MatrixViewPackOnQuery, MatrixTemplated> {
    MatrixViewPackOnQuery(int64_t l_x, int64_t l_y)
        : MatrixViewRowMajor(l_x, l_y) {
    }

public:
    MatrixViewPackOnQuery(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static MatrixViewPackOnQuery allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return MatrixViewPackOnQuery(l_x, l_y);
    }
};

class A1 : public internal::MatrixViewColumnMajor, public internal::QueryView<A1, A1> {
    A1(int64_t l_x, int64_t l_y)
        : MatrixViewColumnMajor(l_x, l_y) {
    }

public:
    A1(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewColumnMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static A1 allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return A1(l_x, l_y);
    }
};

template<int64_t L_X, int64_t L_Y>
class A2 : public internal::MatrixStaticPackedForKernelLeft<L_X, L_Y>, public internal::QueryView<A2<L_X, L_Y>, A1> {
public:
    A2() = default;
    void destroy() {
    }
    static A2 allocate(int64_t, int64_t, int64_t, int64_t) {
        return A2();
    }
};

class A3 : public internal::MatrixViewRowMajor, public internal::QueryStaticMem<A3, A2> {
    A3(int64_t l_x, int64_t l_y)
        : MatrixViewRowMajor(l_x, l_y) {
    }

public:
    A3(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static A3 allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return A3(l_x, l_y);
    }

    // template <int64_t L_X, int64_t L_Y>
    // A2<L_X, L_Y> query(int64_t x, int64_t y) {
    //     return A2<L_X, L_Y>{};
    // }
};

class B1 : public internal::MatrixViewRowMajor, public internal::QueryView<B1, B1> {
    B1(int64_t l_x, int64_t l_y)
        : MatrixViewRowMajor(l_x, l_y) {
    }

public:
    B1(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static B1 allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return B1(l_x, l_y);
    }
};

template<int64_t L_X, int64_t L_Y>
class B2 : public internal::MatrixStaticPackedForKernelRight<L_X, L_Y>, public internal::QueryView<B2<L_X, L_Y>, B1> {
public:
    B2() = default;
    void destroy() {
    }
    static B2 allocate(int64_t, int64_t, int64_t, int64_t) {
        return B2();
    }
};

class B3 : public internal::MatrixViewRowMajor, public internal::QueryStaticMem<B3, B2> {
    B3(int64_t l_x, int64_t l_y)
        : MatrixViewRowMajor(l_x, l_y) {
    }

public:
    B3(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static B3 allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return B3(l_x, l_y);
    }
};

template<typename T>
void random_fill(T &a) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    for (int i = 0; i < a.row; i++) {
        for (int j = 0; j < a.col; j++) {
            a.get(i, j) = dis(gen);
        }
    }
}

template<typename T>
void fill_with(T &a, float val) {
    for (int i = 0; i < a.row; i++) {
        for (int j = 0; j < a.col; j++) {
            a.get(i, j) = val;
        }
    }
}

template<typename T1, typename T2>
void copy_data(T1 &from, T2 &to) {
    for (int i = 0; i < from.row; i++) {
        for (int j = 0; j < from.col; j++) {
            to.get(i, j) = from.get(i, j);
        }
    }
}

// This is the actual implementation of the inner kernel, for 8 by 16 C
template<typename T1, typename T2, typename T3>
void inner_kernel(T1 &a, T2 &b, T3 &c, int64_t) {
    // int64_t N = c.row, M = c.col, K = a.col;
    // for (int64_t i = 0; i < N; ++i)
    // for (int64_t j = 0; j < M; ++j)
    // for (int64_t k = 0; k < K; ++k)
    //     c.get(i, j) += a.get(i, k) * b.get(k, j);

    int64_t K = a.col;

    __m256 mB0;  // __m256 means 256-bit wide. This is introduced in AVX2 (AVX-512, in 2015, has 512, etc.)
    __m256 mB1;
    __m256 mA0;
    __m256 mA1;

    __m256 result0_0 = _mm256_set1_ps(0);  // Broadcast 32-bit (SP) 0 to all 8 elements
    __m256 result1_0 = _mm256_set1_ps(0);
    __m256 result2_0 = _mm256_set1_ps(0);
    __m256 result3_0 = _mm256_set1_ps(0);
    __m256 result4_0 = _mm256_set1_ps(0);
    __m256 result5_0 = _mm256_set1_ps(0);
    __m256 result6_0 = _mm256_set1_ps(0);
    __m256 result7_0 = _mm256_set1_ps(0);

    __m256 result0_1 = _mm256_set1_ps(0);
    __m256 result1_1 = _mm256_set1_ps(0);
    __m256 result2_1 = _mm256_set1_ps(0);
    __m256 result3_1 = _mm256_set1_ps(0);
    __m256 result4_1 = _mm256_set1_ps(0);
    __m256 result5_1 = _mm256_set1_ps(0);
    __m256 result6_1 = _mm256_set1_ps(0);
    __m256 result7_1 = _mm256_set1_ps(0);

    // This is the same for loop as in naive implementation, except now instead of the k indexing
    // a single dot product of 2 vectors of size k (a row of A and a col of B),
    // the k is indexing 6 rows of A and 16 cols of B
    // Since the SIMD width is 8 (256 bits), need to do 12 fmas here
    for (int k = 0; k < K; ++k) {

        // Prefetch k+1'th row of B. Gives ~10% speedup
        __builtin_prefetch(&b.get(k + 1, 0));
        __builtin_prefetch(&b.get(k + 1, 8));

        // Load the k'th row of the B block (load twice since in total, it's 16 floats)
        mB0 = _mm256_load_ps(&b.get(k, 0));
        mB1 = _mm256_load_ps(&b.get(k, 8));

        // Load a single value for the k'th col of A
        // In total, we need to do this 6 times (col of A has height 6)
        // Note: the addresses below must be aligned on a 32-byte boundary
        mA0 = _mm256_set1_ps(a.get(0, k));
        mA1 = _mm256_set1_ps(a.get(1, k));

        // Now we have the 16 floats of B in mB0|mB1, and the 2 floats
        // of A broadcast in mA0 and mA1.
        result0_0 = _mm256_fmadd_ps(mB0, mA0, result0_0);  // result = arg1 .* arg2 .+ arg3
        result0_1 = _mm256_fmadd_ps(mB1, mA0, result0_1);
        result1_0 = _mm256_fmadd_ps(mB0, mA1, result1_0);
        result1_1 = _mm256_fmadd_ps(mB1, mA1, result1_1);
        // result0_0 now contains the final result, for this k,
        // of row 0 and cols 0-7.
        // result0_1 now contains the final result, for this k,
        // of row 0 and cols 8-15.
        // result1_0 now contains the final result, for this k,
        // of row 1 and cols 0-7.
        // result1_1 now contains the final result, for this k,
        // of row 1 and cols 8-15.

        // Repeat for the other 4

        mA0 = _mm256_set1_ps(a.get(2, k));
        mA1 = _mm256_set1_ps(a.get(3, k));
        result2_0 = _mm256_fmadd_ps(mB0, mA0, result2_0);
        result2_1 = _mm256_fmadd_ps(mB1, mA0, result2_1);
        result3_0 = _mm256_fmadd_ps(mB0, mA1, result3_0);
        result3_1 = _mm256_fmadd_ps(mB1, mA1, result3_1);

        mA0 = _mm256_set1_ps(a.get(4, k));
        mA1 = _mm256_set1_ps(a.get(5, k));
        result4_0 = _mm256_fmadd_ps(mB0, mA0, result4_0);
        result4_1 = _mm256_fmadd_ps(mB1, mA0, result4_1);
        result5_0 = _mm256_fmadd_ps(mB0, mA1, result5_0);
        result5_1 = _mm256_fmadd_ps(mB1, mA1, result5_1);

        mA0 = _mm256_set1_ps(a.get(6, k));
        mA1 = _mm256_set1_ps(a.get(7, k));
        result6_0 = _mm256_fmadd_ps(mB0, mA0, result6_0);
        result6_1 = _mm256_fmadd_ps(mB1, mA0, result6_1);
        result7_0 = _mm256_fmadd_ps(mB0, mA1, result7_0);
        result7_1 = _mm256_fmadd_ps(mB1, mA1, result7_1);
    }

    *((__m256 *)(&c.get(0, 0))) += result0_0;
    *((__m256 *)(&c.get(0, 8))) += result0_1;
    *((__m256 *)(&c.get(1, 0))) += result1_0;
    *((__m256 *)(&c.get(1, 8))) += result1_1;
    *((__m256 *)(&c.get(2, 0))) += result2_0;
    *((__m256 *)(&c.get(2, 8))) += result2_1;
    *((__m256 *)(&c.get(3, 0))) += result3_0;
    *((__m256 *)(&c.get(3, 8))) += result3_1;
    *((__m256 *)(&c.get(4, 0))) += result4_0;
    *((__m256 *)(&c.get(4, 8))) += result4_1;
    *((__m256 *)(&c.get(5, 0))) += result5_0;
    *((__m256 *)(&c.get(5, 8))) += result5_1;
    *((__m256 *)(&c.get(6, 0))) += result6_0;
    *((__m256 *)(&c.get(6, 8))) += result6_1;
    *((__m256 *)(&c.get(7, 0))) += result7_0;
    *((__m256 *)(&c.get(7, 8))) += result7_1;
}

#ifdef BLIS_KERNEL
template<typename T1, typename T2, typename T3>
void blis_kernel(T1 &a, T2 &b, T3 &c, int64_t) {
    // int64_t N = c.row, M = c.col, K = a.col;
    // for (int64_t i = 0; i < N; ++i)
    // for (int64_t j = 0; j < M; ++j)
    // for (int64_t k = 0; k < K; ++k)
    //     c.get(i, j) += a.get(i, k) * b.get(k, j);

    int64_t K = a.col;
    float one = 1;
    bli_sgemm_skx_asm_32x12_l2(
        32,
        12,
        K,
        &one,
        a.data,
        b.data,
        &one,
        c.data,
        c.lda,
        1,
        nullptr,
        nullptr);
}
#endif

}  // namespace impl
}  // namespace gern