#pragma once

#include <cstring>
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

    float *data;
    int64_t row;
    int64_t col;
    int64_t lda;
};

inline void add(MatrixCPU a, MatrixCPU b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);
        for (int64_t j = 0; j < a.col; j++) {
            b_data[j] += a_data[j];
        }
    }
}

}  // namespace impl
}  // namespace gern