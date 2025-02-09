#pragma once

#include "library/array/impl/cpu-array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdlib.h>

namespace gern {
namespace impl {

/**
 * @brief An MatrixCPU class to test out Gern's
 * codegen facilities.
 *
 */

class MatrixCPU {
public:
    MatrixCPU(float *data, int64_t row, int64_t col, int64_t lda)
        : data(data), row(row), col(col), lda(lda) {
    }

    MatrixCPU(int64_t row, int64_t col, int64_t lda)
        : MatrixCPU((float *)malloc(sizeof(float) * row * col), row, col, lda) {
    }

    static MatrixCPU allocate(int64_t, int64_t, int64_t l_x, int64_t l_y) {
        return MatrixCPU(l_x, l_y, l_x);
    }

    MatrixCPU query(int64_t x, int64_t y, int64_t l_x, int64_t l_y) {
        return MatrixCPU(data + (x * lda + y), l_x, l_y, lda);
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

    void random_fill(float min = 0.0f, float max = 1.0f) {
        float *data_tmp;
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dist(min, max);
        for (int64_t i = 0; i < row; i++) {
            data_tmp = data + (i * lda);
            for (int64_t j = 0; j < col; j++) {
                data_tmp[j] = (float)dist(gen);
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
    os << "[" << "\n";
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

inline void add(MatrixCPU a, MatrixCPU b) {
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

inline void divn(MatrixCPU a, float n, MatrixCPU b) {
	float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);
        for (int64_t j = 0; j < a.col; j++) {
            b_data[j] = a_data[j] / n;
        }
    }
}

inline void transpose(MatrixCPU a, MatrixCPU b) {
	float *a_data;
	float *b_data_j_i;	
	for (int64_t i = 0; i < a.row; i++) {
		a_data = a.data + (i * a.lda);
		for (int64_t j = 0; j < a.col; j++) {
			b_data_j_i = b.data + (j * b.lda) + i;
			*b_data_j_i = a_data[j];
		}
	}
}

inline void softmax(MatrixCPU a, MatrixCPU b) {
	float *a_data;
	float *b_data;
	for (int64_t i = 0; i < a.row; i++) {
		a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);	
		float exp_sum = 0;
		for (int64_t j = 0; j < a.col; j++) {
			exp_sum += expf(a_data[j]);
		}
		for (int64_t j = 0; j < a.col; j++) {
			b_data[j] = expf(a_data[j]) / exp_sum;
		}
	}
}

inline void mmul(MatrixCPU a, MatrixCPU b, MatrixCPU out, int64_t shared_len) {
	for (int i = 0; i < a.row; i++) {
		for (int j = 0; j < b.col; j++) {
			float acc = 0;
			for (int k = 0; k < a.col; k++) {
				acc += *(a.data + (i * a.lda) + k) * *(b.data + (k * b.lda) + j);
			}
			*(out.data + (i * out.lda) + j) = acc;
		}
    }
}

inline void exp_matrix(MatrixCPU a, MatrixCPU b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);
        for (int64_t j = 0; j < a.col; j++) {
            b_data[j] = expf(a_data[j]);
        }
    }
}

inline void sum_row(MatrixCPU a, ArrayCPU b) {
    float *a_data;
    for (int64_t i = 0; i < a.row; i++) {
        float sum = 0.0f;
        a_data = a.data + (i * a.lda);
        for (int64_t j = 0; j < a.col; j++) {
            sum += a_data[j];
        }
        b.data[i] = sum;
    }
}

inline void max_row(MatrixCPU a, ArrayCPU b) {
    float *a_data;
    for (int64_t i = 0; i < a.row; i++) {
        float maximum = std::numeric_limits<float>::min();
        a_data = a.data + (i * a.lda);
        for (int64_t j = 0; j < a.col; j++) {
            maximum = std::max(maximum, a_data[j]);
        }
        b.data[i] = maximum;
    }
}

inline void subtract_vec(ArrayCPU b, MatrixCPU a, MatrixCPU out) {
    float *a_data;
    float *out_data;
    for (int64_t i = 0; i < a.row; i++) {
        float vec_data = b.data[i];
        a_data = a.data + (i * a.lda);
        out_data = out.data + (i * out.lda);
        for (int64_t j = 0; j < a.col; j++) {
            out_data[j] = vec_data - a_data[j];
        }
    }
}

inline void divide_vec(ArrayCPU b, MatrixCPU a, MatrixCPU out) {
    float *a_data;
    float *out_data;
    for (int64_t i = 0; i < a.row; i++) {
        float vec_data = b.data[i];
        a_data = a.data + (i * a.lda);
        out_data = out.data + (i * out.lda);
        for (int64_t j = 0; j < a.col; j++) {
            out_data[j] = a_data[j] / vec_data;
        }
    }
}

}  // namespace impl
}  // namespace gern