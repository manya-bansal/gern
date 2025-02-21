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
    : m_data(new float[row * col]), m_lda(col), row(row), col(col) {
    }
    MatrixBase() = delete;
public:
    int64_t row;
    int64_t col;
    
    float *data() const {
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
        delete[] this->m_data;
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
    MatrixTemplated() = default;
    float m_data[L_X * L_Y];
public:

    int64_t row = L_X;
    int64_t col = L_Y;

    float *data() const {
        return m_data;
    }

    static MatrixTemplated allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
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
        MatrixTemplated<L_X, L_Y> res{};
        for (int64_t i = 0; i < L_X; ++i) {
            std::copy(
                &this->get(x + i, y),
                &this->get(x + i, y + L_Y),
                &res->get(i, 0));
        }
        return res;
    }

    template<int64_t L_X, int64_t L_Y>
    void insert(int64_t x, int64_t y, MatrixTemplated<L_X, L_Y> to_insert) {
        for (int64_t i = 0; i < L_X; ++i) {
            std::copy(
                &to_insert->get(i, 0),
                &to_insert->get(i, L_Y),
                &this->get(x + i, y));
        }
    }
};

template <MatrixType T>
void matrix_multiply(T &a, T &b, T &c, int64_t k_dim) {
    int64_t N = c.row, M = c.col, K = k_dim;
    
    for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < M; ++j)
    for (int64_t k = 0; k < K; ++k)
        c.get(i, j) += a.get(i, k)* b.get(k, j);

}

}  // namespace impl
}  // namespace gern