#pragma once

#include "library/array/impl/cpu-array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdlib.h>
// #include <Accelerate/Accelerate.h>

namespace gern {
namespace impl {

/**
 * @brief An MatrixCPU class to test out Gern's
 * codegen facilities.
 *
 */

template<int Rows, int Cols>
struct MatrixCPUStatic {
    float data[Rows][Cols] = {0};
};

class MatrixCPUTemplate {
public:
	MatrixCPUTemplate(float *data, int64_t row, int64_t col)
		: data(data), row(row), col(col) {
	}

	MatrixCPUTemplate(int64_t row, int64_t col)
		: MatrixCPUTemplate((float *)malloc(sizeof(float) * row * col), row, col) {
	}

	template <int64_t NumRows, int64_t NumCols>
    MatrixCPUStatic<NumRows, NumCols> query(int64_t x, int64_t y) {
		MatrixCPUStatic<NumRows, NumCols> queried;
		for (int i = 0; i < NumRows; i++) {
			for (int j = 0; j < NumCols; j++) {
				queried.data[i][j] = data[(x + i) * col + y + j];
			}
		}
        return queried; 
    }

	template <int64_t RowsToInsert, int64_t ColsToInsert>
    void insert(int64_t x, int64_t y, MatrixCPUStatic<RowsToInsert, ColsToInsert>  to_insert) {
		for (int i = 0; i < RowsToInsert; i++) {
			for (int j = 0; j < ColsToInsert; j++) {
				data[(x + i) * col + y + j] = to_insert.data[i][j];
			}
		}
    }

    void destroy() {
    }

    void vvals(float f) {
        float *data_tmp;
        for (int64_t i = 0; i < row; i++) {
            data_tmp = data + (i * col);
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
            data_tmp = data + (i * col);
            for (int64_t j = 0; j < col; j++) {
                data_tmp[j] = (float)dist(gen);
            }
        }
    }

    void ascending() {
        float *data_tmp;
        for (int64_t i = 0; i < row; i++) {
            data_tmp = data + (i * col);
            for (int64_t j = 0; j < col; j++) {
                data_tmp[j] = i * col + j;
            }
        }
    }

    float *data;
    int64_t row;
    int64_t col;
};

template <int64_t Rows, int64_t Cols>
[[maybe_unused]] static std::ostream &operator<<(std::ostream &os, const MatrixCPUStatic<Rows, Cols> &m) {
    os << "["
       << "\n";
    for (int64_t i = 0; i < Rows; i++) {
        for (int64_t j = 0; j < Cols; j++) {
            os << m.data[i][j] << " ";
        }
        os << "\n";
    }
    os << "]";
    return os;
}

[[maybe_unused]] static std::ostream &operator<<(std::ostream &os, const MatrixCPUTemplate &m) {
    float *data_tmp;
    os << "["
       << "\n";
    for (int64_t i = 0; i < m.row; i++) {
        data_tmp = m.data + (i * m.col);
        for (int64_t j = 0; j < m.col; j++) {
            os << data_tmp[j] << " ";
        }
        os << "\n";
    }
    os << "]";
    return os;
}

template <int Rows, int Cols>
inline void add(MatrixCPUStatic<Rows, Cols> &a, MatrixCPUStatic<Rows, Cols> &b) {
    for (int64_t i = 0; i < Rows; i++) {
        for (int64_t j = 0; j < Cols; j++) {
            b.data[i][j] = a.data[i][j] + 1;
        }
    }
}


template<int Rows, int Cols>
MatrixCPUStatic<Rows, Cols> temp_matrix_allocate() {
    return MatrixCPUStatic<Rows, Cols>();
}

// inline void add(MatrixCPU a, MatrixCPU b) {
//     float *a_data;
//     float *b_data;
//     for (int64_t i = 0; i < a.row; i++) {
//         a_data = a.data + (i * a.lda);
//         b_data = b.data + (i * b.lda);
//         for (int64_t j = 0; j < a.col; j++) {
//             b_data[j] = a_data[j] + 1;
//         }
//     }
// }

// inline void divn(MatrixCPU a, float n, MatrixCPU b) {
// 	float *a_data;
//     float *b_data;
//     for (int64_t i = 0; i < a.row; i++) {
//         a_data = a.data + (i * a.lda);
//         b_data = b.data + (i * b.lda);

// 		cblas_scopy(a.col, a_data, 1, b_data, 1);
// 		cblas_sscal(a.col, 1 / n, b_data, 1);
//     }
// }

// inline void transpose(MatrixCPU a, MatrixCPU b) {
// 	float *a_data;
// 	float *b_data_j_i;	
// 	for (int64_t i = 0; i < a.row; i++) {
// 		a_data = a.data + (i * a.lda);
// 		for (int64_t j = 0; j < a.col; j++) {
// 			b_data_j_i = b.data + (j * b.lda) + i;
// 			*b_data_j_i = a_data[j];
// 		}
// 	}
// }


// inline void softmax(MatrixCPU a, MatrixCPU b) {
// 	float *a_data;
// 	float *b_data;

// 	for (int64_t i = 0; i < a.row; i++) {
// 		a_data = a.data + (i * a.lda);
//         b_data = b.data + (i * b.lda);	

// 		int size = a.col;
// 		vvexpf(b_data, a_data, &size);

// 		float exp_sum = 0;

// 		vDSP_sve(b_data, 1, &exp_sum, a.col);
// 		cblas_sscal(a.col, 1 / exp_sum, b_data, 1);
// 	}
// }

// inline void mmul(MatrixCPU a, MatrixCPU b, MatrixCPU out) {
// 	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.row, b.col, a.col, 1, a.data, a.lda, b.data, b.lda, 0, out.data, out.lda);
// }

// inline void exp_matrix(MatrixCPU a, MatrixCPU b) {
// 	float *a_data;
//     float *b_data;
// 	for (int64_t i = 0; i < a.row; i++) {
// 		a_data = a.data + (i * a.lda);
// 		b_data = b.data + (i * b.lda);
// 		int size = a.col;
// 		vvexpf(b_data, a_data, &size);
// 	}
// }

// inline void sum_row(MatrixCPU a, ArrayCPU b) {
//     float *a_data;
//     for (int64_t i = 0; i < a.row; i++) {
//         float sum = 0.0f;
//         a_data = a.data + (i * a.lda);
//         for (int64_t j = 0; j < a.col; j++) {
//             sum += a_data[j];
//         }
//         b.data[i] = sum;
//     }
// }

// inline void max_row(MatrixCPU a, ArrayCPU b) {
//     float *a_data;
//     for (int64_t i = 0; i < a.row; i++) {
//         float maximum = std::numeric_limits<float>::min();
//         a_data = a.data + (i * a.lda);
//         for (int64_t j = 0; j < a.col; j++) {
//             maximum = std::max(maximum, a_data[j]);
//         }
//         b.data[i] = maximum;
//     }
// }

// inline void subtract_vec(ArrayCPU b, MatrixCPU a, MatrixCPU out) {
//     float *a_data;
//     float *out_data;
//     for (int64_t i = 0; i < a.row; i++) {
//         float vec_data = b.data[i];
//         a_data = a.data + (i * a.lda);
//         out_data = out.data + (i * out.lda);
//         for (int64_t j = 0; j < a.col; j++) {
//             out_data[j] = vec_data - a_data[j];
//         }
//     }
// }

// inline void divide_vec(ArrayCPU b, MatrixCPU a, MatrixCPU out) {
//     float *a_data;
//     float *out_data;
//     for (int64_t i = 0; i < a.row; i++) {
//         float vec_data = b.data[i];
//         a_data = a.data + (i * a.lda);
//         out_data = out.data + (i * out.lda);
//         for (int64_t j = 0; j < a.col; j++) {
//             out_data[j] = a_data[j] / vec_data;
//         }
//     }
// }

// inline void matrix_multiply(MatrixCPU a, MatrixCPU b, MatrixCPU c) {
//     float *a_data;
//     float *b_data;
//     float *c_data;
//     for (int64_t i = 0; i < c.row; i++) {
//         c_data = c.data + (i * c.lda);
//         a_data = a.data + (i * a.lda);
//         for (int64_t j = 0; j < c.col; j++) {
//             float sum = 0.0f;
//             for (int64_t k = 0; k < a.col; k++) {
//                 b_data = b.data + (k * b.lda) + j;
//                 sum += a_data[k] * b_data[0];
//             }
//             c_data[j] += sum;
//         }
//     }
// }

}  // namespace impl
}  // namespace gern