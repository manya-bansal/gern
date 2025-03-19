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

namespace gern {
namespace impl {

namespace internal {
constexpr size_t matrix_alignment = 64;

enum class Layout {
    RowMajor,
    ColumnMajor
};

class MatrixViewBase {
protected:
    MatrixViewBase(float *data, int64_t x, int64_t y, int64_t lda)
        : data(data), row(x), col(y), lda(lda) {
    }

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
    MatrixViewRowMajor(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewBase(data, x, y, lda) {
    }
    float &get(int64_t x, int64_t y) {
        return this->data[x * this->lda + y];
    }
};

class MatrixViewColumnMajor : public MatrixViewBase {
public:
    static constexpr Layout layout = Layout::ColumnMajor;
    MatrixViewColumnMajor(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewBase(data, x, y, lda) {
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
class MatrixStaticMemColumnMajor {
protected:
    static constexpr int64_t matrix_alignment_in_floats = matrix_alignment / sizeof(float);

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
        for (int i = 0; i < L_X; i++) {
            for (int j = 0; j < L_Y; j++) {
                res.get(i, j) = static_cast<Derived *>(this)->get(x + i, y + j);
            }
        }
        return res;
    }
};

}  // namespace internal

// Here are the actual types that will be used
class MatrixViewBasic : public internal::MatrixViewRowMajor, public internal::QueryView<MatrixViewBasic, MatrixViewBasic> {
public:
    MatrixViewBasic(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static MatrixViewBasic allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        constexpr int64_t matrix_alignment_in_floats = internal::matrix_alignment / sizeof(float);
        int lda = (l_y + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
        return MatrixViewBasic((float *)aligned_alloc(internal::matrix_alignment, l_x * lda * sizeof(float)), l_x, l_y, lda);
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
public:
    MatrixViewPackOnQuery(float *data, int64_t x, int64_t y, int64_t lda)
        : MatrixViewRowMajor(data, x, y, lda) {
    }
    void destroy() {
        free(this->data);
    }
    static MatrixViewPackOnQuery allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        constexpr int64_t matrix_alignment_in_floats = internal::matrix_alignment / sizeof(float);
        int lda = (l_y + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
        return MatrixViewPackOnQuery((float *)aligned_alloc(internal::matrix_alignment, l_x * lda * sizeof(float)), l_x, l_y, lda);
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

// This is the actual implementation of the inner kernel, for 8 by 16 C
template<typename T1, typename T2, typename T3>
void inner_kernel(T1 &a, T2 &b, T3 &c, int64_t) {
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

}  // namespace impl
}  // namespace gern