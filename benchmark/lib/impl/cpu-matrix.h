#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <stdlib.h>
#include <concepts>

namespace gern {
namespace impl {

constexpr size_t matrix_alignment = 64;

template <typename T>
concept MatrixType = requires(T obj, int64_t x, int64_t y) {
    {obj.row} -> std::convertible_to<int64_t>;
    {obj.col} -> std::convertible_to<int64_t>;
    {obj.get(x, y)} -> std::convertible_to<float &>;
    {obj.data()} -> std::convertible_to<float *>;
    {obj.vvals(0)} -> std::same_as<void>;
};


class MatrixBase {
protected:
    float *m_data;
    int64_t m_lda;
    MatrixBase(float *data, int64_t row, int64_t col, int64_t lda)
    : m_data(data),  m_lda(lda), row(row), col(col) {
    }
    MatrixBase(int64_t row, int64_t col)
    : row(row), col(col) {
        
        constexpr int64_t matrix_alignment_in_floats = matrix_alignment / sizeof(float);
        this->m_lda = (this->col + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;
        this->m_data = (float *)aligned_alloc(
            matrix_alignment, this->row * this->m_lda * sizeof(float)
        );
    }
    MatrixBase() = delete;
public:
    int64_t row;
    int64_t col;
    
    float *data() {
        return m_data;
    }

    float &get(int64_t x, int64_t y) {
        return this->data()[x * this->m_lda + y];
    }

    void insert(int64_t x, int64_t y, int64_t l_x, int64_t l_y, MatrixBase to_insert) {
        for (int i = 0; i < l_x; ++i) {
            std::copy(
                &to_insert.get(i, 0),
                &to_insert.get(i, l_y),
                &this->get(x + i, y));
        }
    }

    void destroy() {
        free(this->m_data);
    }

    void vvals(float f) {
        for (int i = 0; i < this->row; ++i) {
            for (int j = 0; j < this->col; ++j) {
                this->get(i, j) = f;
            }
        }
    }
    void random_fill(float min = 0.0f, float max = 1.0f) {
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dist(min, max);
        for (int i = 0; i < this->row; ++i) {
            for (int j = 0; j < this->col; ++j) {
                this->get(i, j) = dist(gen);
            }
        }
    }
};

template<int64_t L_X, int64_t L_Y>
class MatrixTemplated {
private:

    static constexpr int64_t matrix_alignment_in_floats = matrix_alignment / sizeof(float);
    static constexpr int64_t LDA = (L_Y + matrix_alignment_in_floats - 1) / matrix_alignment_in_floats * matrix_alignment_in_floats;

    MatrixTemplated() = default;
    alignas(matrix_alignment) float m_data[L_X * LDA];
public:

    int64_t row = L_X;
    int64_t col = L_Y;

    float *data() {
        return m_data;
    }

    static MatrixTemplated allocate(int64_t, int64_t, int64_t, int64_t) {
        return MatrixTemplated();
    }
    float &get(int64_t x, int64_t y) {
        return this->data()[x * L_Y + y];
    }

    template<int64_t Q_X, int64_t Q_Y>
    MatrixTemplated<Q_X, Q_Y> query(int64_t x, int64_t y) {
        MatrixTemplated<Q_X, Q_Y> res;
        for (int i = 0; i < Q_X; ++i) {
            std::copy(
                &this->get(x + i, y),
                &this->get(x + i, y + Q_Y),
                &res.get(i, 0));
        }
        return res;
    }

    template<int64_t Q_X, int64_t Q_Y>
    void insert(int64_t x, int64_t y, MatrixTemplated<Q_X, Q_Y> to_insert) {
        for (int i = 0; i < Q_X; ++i) {
            std::copy(
                &to_insert.get(i, 0),
                &to_insert.get(i, Q_Y),
                &this->get(x + i, y));
        }
    }

    void destroy() {
    }
    void vvals(float f) {
        for (int i = 0; i < L_X; ++i) {
            for (int j = 0; j < L_Y; ++j) {
                this->get(i, j) = f;
            }
        }
    }
};

class MatrixViewBasic : public MatrixBase {
private:
    MatrixViewBasic(int64_t row, int64_t col)
        : MatrixBase(row, col) {
    }
    MatrixViewBasic() = delete;

public:
    static MatrixViewBasic allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return MatrixViewBasic(l_x, l_y);
    }
    MatrixViewBasic(float *data, int64_t row, int64_t col, int64_t lda)
        : MatrixBase(data, row, col, lda) {
    }
    MatrixViewBasic query(int64_t x, int64_t y, int64_t l_x, int64_t l_y) {
        return MatrixViewBasic(&this->get(x, y), l_x, l_y, this->m_lda);
    }
};

class MatrixViewPackOnQuery : public MatrixBase {
private:
    MatrixViewPackOnQuery(int64_t row, int64_t col)
        : MatrixBase(row, col) {
    }
    MatrixViewPackOnQuery() = delete;

public:
    static MatrixViewPackOnQuery allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return MatrixViewPackOnQuery(l_x, l_y);
    }
    MatrixViewPackOnQuery(float *data, int64_t row, int64_t col, int64_t lda)
        : MatrixBase(data, row, col, lda) {
    }

    template<int64_t L_X, int64_t L_Y>
    MatrixTemplated<L_X, L_Y> query(int64_t x, int64_t y) {
        auto res = MatrixTemplated<L_X, L_Y>::allocate(0, 0, 0, 0);
        for (int64_t i = 0; i < L_X; ++i) {
            std::copy(
                &this->get(x + i, y),
                &this->get(x + i, y + L_Y),
                &res.get(i, 0));
        }
        return res;
    }

    template<int64_t L_X, int64_t L_Y>
    void insert(int64_t x, int64_t y, MatrixTemplated<L_X, L_Y> to_insert) {
        for (int64_t i = 0; i < L_X; ++i) {
            std::copy(
                &to_insert.get(i, 0),
                &to_insert.get(i, L_Y),
                &this->get(x + i, y));
        }
    }
};

template <int64_t N, int64_t M, int64_t K, typename T>
void matrix_multiply_templated(MatrixTemplated<N, K> &a, MatrixTemplated<K, M> &b, T &c, int64_t) {    
    for (int64_t i = 0; i < N; ++i)
    for (int64_t k = 0; k < K; ++k)
    for (int64_t j = 0; j < M; ++j)
        c.get(i, j) += a.get(i, k)* b.get(k, j);
}

template <MatrixType T1, MatrixType T2, MatrixType T3>
void matrix_multiply_generic(T1 &a, T2 &b, T3 &c, int64_t k_dim) {
    int64_t N = c.row, M = c.col, K = k_dim;
    
    for (int64_t k = 0; k < K; ++k)
    for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < M; ++j)
        c.get(i, j) += a.get(i, k)* b.get(k, j);
}


// inner kernel for 8 by 16 matrix
template <MatrixType T1, MatrixType T2, MatrixType T3>
void inner_kernel(T1 &a, T2 &b, T3 &c, int64_t) {
    int64_t K = a.col;

    __m256 mB0; // __m256 means 256-bit wide. This is introduced in AVX2 (AVX-512, in 2015, has 512, etc.)
    __m256 mB1;
    __m256 mA0;
    __m256 mA1;

    // Chose kernel size 6x16
    // - 16 because SIMD width is 8*32 (so must be multiple of 8)
    // - Also overall 16 registers
    // - Number of registers depends on AVX, AVX-2 or AVX-512
    // - So having 6x16 means 6x2 registers used for C block
    // - This leaves 4 for sections of A and B (needed to do fma)
    // - To use SIMD, need to store in registers
    // - Note: Intel paper uses 30x8, not 6x16
    __m256 result0_0  = _mm256_set1_ps(0); // Broadcast 32-bit (SP) 0 to all 8 elements
    __m256 result1_0  = _mm256_set1_ps(0);
    __m256 result2_0  = _mm256_set1_ps(0);
    __m256 result3_0  = _mm256_set1_ps(0);
    __m256 result4_0  = _mm256_set1_ps(0);
    __m256 result5_0  = _mm256_set1_ps(0);
    __m256 result6_0  = _mm256_set1_ps(0);
    __m256 result7_0  = _mm256_set1_ps(0);
    
    __m256 result0_1  = _mm256_set1_ps(0);
    __m256 result1_1  = _mm256_set1_ps(0);
    __m256 result2_1  = _mm256_set1_ps(0);
    __m256 result3_1  = _mm256_set1_ps(0);
    __m256 result4_1  = _mm256_set1_ps(0);
    __m256 result5_1  = _mm256_set1_ps(0);
    __m256 result6_1  = _mm256_set1_ps(0);
    __m256 result7_1  = _mm256_set1_ps(0);

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing 6 rows of A and 16 cols of B
	// Since the SIMD width is 8 (256 bits), need to do 12 fmas here
    for(int k=0; k<K; ++k) {
        
        // Prefetch k+1'th row of B. Gives ~10% speedup
        __builtin_prefetch(&b.get(k+1, 0));        
        __builtin_prefetch(&b.get(k+1, 8));
        
		// Load the k'th row of the B block (load twice since in total, it's 16 floats)
        mB0   = _mm256_load_ps(&b.get(k, 0)); 
        mB1   = _mm256_load_ps(&b.get(k, 8));
        
        // Load a single value for the k'th col of A
        // In total, we need to do this 6 times (col of A has height 6)
        // Note: the addresses below must be aligned on a 32-byte boundary
        mA0 = _mm256_set1_ps(a.get(0, k));
        mA1 = _mm256_set1_ps(a.get(1, k));

        
        // Now we have the 16 floats of B in mB0|mB1, and the 2 floats
        // of A broadcast in mA0 and mA1.
        result0_0      = _mm256_fmadd_ps(mB0,mA0,result0_0); // result = arg1 .* arg2 .+ arg3
        result0_1      = _mm256_fmadd_ps(mB1,mA0,result0_1);
        result1_0      = _mm256_fmadd_ps(mB0,mA1,result1_0);
        result1_1      = _mm256_fmadd_ps(mB1,mA1,result1_1);
        // result0_0 now contains the final result, for this k,
        // of row 0 and cols 0-7.
        // result0_1 now contains the final result, for this k,
        // of row 0 and cols 8-15.
        // result1_0 now contains the final result, for this k,
        // of row 1 and cols 0-7.
        // result1_1 now contains the final result, for this k,
        // of row 1 and cols 8-15.
        
        // Repeat for the other 4
        
        mA0   = _mm256_set1_ps(a.get(2, k));
        mA1   = _mm256_set1_ps(a.get(3, k));
        result2_0      = _mm256_fmadd_ps(mB0,mA0,result2_0);
        result2_1      = _mm256_fmadd_ps(mB1,mA0,result2_1);
        result3_0      = _mm256_fmadd_ps(mB0,mA1,result3_0);
        result3_1      = _mm256_fmadd_ps(mB1,mA1,result3_1);
        
        mA0   = _mm256_set1_ps(a.get(4, k));
        mA1   = _mm256_set1_ps(a.get(5, k));
        result4_0      = _mm256_fmadd_ps(mB0,mA0,result4_0);
        result4_1      = _mm256_fmadd_ps(mB1,mA0,result4_1);
        result5_0      = _mm256_fmadd_ps(mB0,mA1,result5_0);
        result5_1      = _mm256_fmadd_ps(mB1,mA1,result5_1);


        mA0   = _mm256_set1_ps(a.get(6, k));
        mA1   = _mm256_set1_ps(a.get(7, k));
        result6_0      = _mm256_fmadd_ps(mB0,mA0,result6_0);
        result6_1      = _mm256_fmadd_ps(mB1,mA0,result6_1);
        result7_0      = _mm256_fmadd_ps(mB0,mA1,result7_0);
        result7_1      = _mm256_fmadd_ps(mB1,mA1,result7_1);
    }

    *((__m256*) (&c.get(0, 0))) += result0_0;
    *((__m256*) (&c.get(0, 8))) += result0_1;
    *((__m256*) (&c.get(1, 0))) += result1_0;
    *((__m256*) (&c.get(1, 8))) += result1_1;
    *((__m256*) (&c.get(2, 0))) += result2_0;
    *((__m256*) (&c.get(2, 8))) += result2_1;
    *((__m256*) (&c.get(3, 0))) += result3_0;
    *((__m256*) (&c.get(3, 8))) += result3_1;
    *((__m256*) (&c.get(4, 0))) += result4_0;
    *((__m256*) (&c.get(4, 8))) += result4_1;
    *((__m256*) (&c.get(5, 0))) += result5_0;
    *((__m256*) (&c.get(5, 8))) += result5_1;
    *((__m256*) (&c.get(6, 0))) += result6_0;
    *((__m256*) (&c.get(6, 8))) += result6_1;
    *((__m256*) (&c.get(7, 0))) += result7_0;
    *((__m256*) (&c.get(7, 8))) += result7_1;   
}



}  // namespace impl
}  // namespace gern