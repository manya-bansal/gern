#pragma once

#include "library/array/impl/cpu-array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <ATen/ATen.h>
#include <torch/torch.h>

namespace gern {
namespace impl {
class MatrixCPU4Dim {
public:
    MatrixCPU4Dim(at::Tensor tensor)
        : tensor(tensor), dims(tensor.sizes().vec()) {}

    MatrixCPU4Dim(float* data_ptr, int64_t w, int64_t x, int64_t y, int64_t z)
        : tensor(torch::from_blob(data_ptr, {w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))),
          dims({w, x, y, z}) {}
    
    MatrixCPU4Dim(int64_t w, int64_t x, int64_t y, int64_t z)
        : MatrixCPU4Dim(torch::empty({w, x, y, z}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))) {}

    static MatrixCPU4Dim allocate(int64_t w, int64_t x, int64_t y, int64_t z) {
        return MatrixCPU4Dim(w, x, y, z);
    }

    void destroy() {
    }

    MatrixCPU4Dim query(int64_t w, int64_t x, int64_t y, int64_t z, int64_t l_w, int64_t l_x, int64_t l_y, int64_t l_z) {
        return MatrixCPU4Dim(tensor.index({torch::indexing::Slice(w, w + l_w),
                                            torch::indexing::Slice(x, x + l_x),
                                            torch::indexing::Slice(y, y + l_y),
                                            torch::indexing::Slice(z, z + l_z)}));
    }

    void insert(int64_t w, int64_t x, int64_t y, int64_t z,
        int64_t l_w, int64_t l_x, int64_t l_y, int64_t l_z,
        const MatrixCPU4Dim& to_insert) {
        tensor.index_put_({torch::indexing::Slice(w, w + l_w),
                torch::indexing::Slice(x, x + l_x),
                torch::indexing::Slice(y, y + l_y),
                torch::indexing::Slice(z, z + l_z)},
                to_insert.tensor);
    }

    void vvals(float value) {
        tensor.fill_(value);
    }

    void random_fill(float min = 0.0f, float max = 1.0f) {
        at::Tensor noise = torch::rand(dims, tensor.options()) * (max - min) + min;
        tensor.copy_(noise);
    }

    void ascending() {
        int64_t total = tensor.numel();
        at::Tensor flat = torch::arange(0, total, tensor.options());
        tensor.copy_(flat.view(dims));
    }

    float* at(int64_t i, int64_t j, int64_t k, int64_t l) {
        return tensor[i][j][k][l].data_ptr<float>();
    }

    at::Tensor tensor;
    std::vector<int64_t> dims;
};
// class MatrixCPU4Dim {
// public:
//     MatrixCPU4Dim(float *data, int64_t i_dim, int64_t j_dim, int64_t k_dim, int64_t l_dim, int64_t i_incr, int64_t j_incr, int64_t k_incr)
//         : data(data), dims({i_dim, j_dim, k_dim, l_dim}), i_incr(i_incr), j_incr(j_incr), k_incr(k_incr) {
//     }

//     MatrixCPU4Dim(int64_t i_dim, int64_t j_dim, int64_t k_dim, int64_t l_dim, int64_t i_incr, int64_t j_incr, int64_t k_incr)
//         : MatrixCPU4Dim((float *)malloc(sizeof(float) * i_dim * j_dim * k_dim * l_dim), i_dim, j_dim, k_dim, l_dim, i_incr, j_incr, k_incr) {
//     }

//     static MatrixCPU4Dim allocate(int64_t, int64_t, int64_t, int64_t, int64_t l_w, int64_t l_x, int64_t l_y, int64_t l_z) {
//         return MatrixCPU4Dim(l_w, l_x, l_y, l_z, l_x, l_y, l_z);
//     }

//     MatrixCPU4Dim query(int64_t w, int64_t x, int64_t y, int64_t z, int64_t l_w, int64_t l_x, int64_t l_y, int64_t l_z) {
//         return MatrixCPU4Dim(data + (w * i_incr * j_incr * k_incr + x * j_incr * k_incr + y * k_incr + z), l_w, l_x, l_y, l_z, i_incr, j_incr, k_incr);
//     }

    // void insert(int64_t w, int64_t x, int64_t y, int64_t z, int64_t l_w, int64_t l_x, int64_t l_y, int64_t l_z, MatrixCPU4Dim to_insert) {
    //     float *data_tmp;
    //     float *data_insert_tmp;
    //     for (int64_t i = 0; i < l_w; i++) {
    //         for (int64_t j = 0; j < l_x; j++) {
    //             for (int64_t k = 0; k < l_y; k++) {
    //                 data_tmp = data + ((w + i) * (i_incr * j_incr * k_incr)) + (x + j) * j_incr * k_incr + (y + k) * k_incr + z;
    //                 data_insert_tmp = to_insert.data + (i * i_incr * j_incr * k_incr) + j * j_incr * k_incr + k * k_incr;
    //                 for (int64_t l = 0; l < l_z; l++) {
    //                     data_tmp[l] = data_insert_tmp[l];
    //                 }
    //             }
    //         }
    //     }
    // }

//     void destroy() {
//         free(data);
//     }

//     void vvals(float f) {
//         float *data_tmp;
//         for (int64_t i = 0; i < dims[0]; i++) {
//             for (int64_t j = 0; j < dims[1]; j++) {
//                 for (int64_t k = 0; k < dims[2]; k++) {
//                     data_tmp = data + (i * i_incr * j_incr * k_incr) + j * j_incr * k_incr + k * k_incr;
//                     for (int64_t l = 0; l < dims[3]; l++) {
//                         data_tmp[l] = f;
//                     }
//                 }
//             }
//         }
//     }

//     void random_fill(float min = 0.0f, float max = 1.0f) {
//         float *data_tmp;
//         std::mt19937 gen(0);
//         std::uniform_real_distribution<float> dist(min, max);
//         for (int64_t i = 0; i < dims[0]; i++) {
//             for (int64_t j = 0; j < dims[1]; j++) {
//                 for (int64_t k = 0; k < dims[2]; k++) {
//                     data_tmp = data + (i * i_incr * j_incr * k_incr) + j * j_incr * k_incr + k * k_incr;
//                     for (int64_t l = 0; l < dims[3]; l++) {
//                         data_tmp[l] = (float)dist(gen);
//                     }
//                 }
//             }
//         }
//     }

//     void ascending() {
//         float *data_tmp;
//         for (int64_t i = 0; i < dims[0]; i++) {
//             for (int64_t j = 0; j < dims[1]; j++) {
//                 for (int64_t k = 0; k < dims[2]; k++) {
//                     data_tmp = data + (i * i_incr * j_incr * k_incr) + j * j_incr * k_incr + k * k_incr;
//                     for (int64_t l = 0; l < dims[3]; l++) {
//                         data_tmp[l] = i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + k * dims[3] + l;
//                     }
//                 }
//             }
//         }
//     }

// 	float *at(int i, int j, int k, int l) {
// 		return &data[i * i_incr * j_incr * k_incr + j * j_incr * k_incr + k * k_incr + l];
// 	}

//     float *data;
// 	std::vector<int64_t> dims;
//     int64_t i_incr;
//     int64_t j_incr;
//     int64_t k_incr;
// };

class MatrixCPU3Dim {
public:
    MatrixCPU3Dim(float *data, int64_t i_dim, int64_t j_dim, int64_t k_dim, int64_t i_incr, int64_t j_incr)
        : data(data), i_dim(i_dim), j_dim(j_dim), k_dim(k_dim), i_incr(i_incr), j_incr(j_incr) {
    }

    MatrixCPU3Dim(int64_t i_dim, int64_t j_dim, int64_t k_dim, int64_t i_incr, int64_t j_incr)
        : MatrixCPU3Dim((float *)malloc(sizeof(float) * i_dim * j_dim * k_dim), i_dim, j_dim, k_dim, i_incr, j_incr) {
    }

    static MatrixCPU3Dim allocate(int64_t, int64_t, int64_t, int64_t l_x, int64_t l_y, int64_t l_z) {
        return MatrixCPU3Dim(l_x, l_y, l_z, l_y, l_z);
    }

    MatrixCPU3Dim query(int64_t x, int64_t y, int64_t z, int64_t l_x, int64_t l_y, int64_t l_z) {
        return MatrixCPU3Dim(data + (x * i_incr * j_incr + y * j_incr + z), l_x, l_y, l_z, i_incr, j_incr);
    }

    void insert(int64_t x, int64_t y, int64_t z, int64_t l_x, int64_t l_y, int64_t l_z, MatrixCPU3Dim to_insert) {
        float *data_tmp;
        float *data_insert_tmp;
        for (int64_t i = 0; i < l_x; i++) {
            for (int64_t j = 0; j < l_y; j++) {
                data_tmp = data + ((x + i) * (i_incr * j_incr)) + (y + j) * j_incr + z;
                data_insert_tmp = to_insert.data + (i * i_incr * j_incr) + j * j_incr;
                for (int64_t k = 0; k < l_z; k++) {
                    data_tmp[k] = data_insert_tmp[k];
                }
            }
        }
    }

    void destroy() {
        free(data);
    }

    void vvals(float f) {
        float *data_tmp;
        for (int64_t i = 0; i < i_dim; i++) {
            for (int64_t j = 0; j < j_dim; j++) {
                data_tmp = data + (i * i_incr * j_incr) + j * j_incr;
                for (int64_t k = 0; k < k_dim; k++) {
                    data_tmp[k] = f;
                }
            }
        }
    }

    void random_fill(float min = 0.0f, float max = 1.0f) {
        float *data_tmp;
        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dist(min, max);
        for (int64_t i = 0; i < i_dim; i++) {
            for (int64_t j = 0; j < j_dim; j++) {
                data_tmp = data + (i * i_incr * j_incr) + j * j_incr;
                for (int64_t k = 0; k < k_dim; k++) {
                    data_tmp[k] = (float)dist(gen);;
                }
            }
        }
    }

    void ascending() {
        float *data_tmp;
        for (int64_t i = 0; i < i_dim; i++) {
            for (int64_t j = 0; j < j_dim; j++) {
                data_tmp = data + (i * i_incr * j_incr) + j * j_incr;
                for (int64_t k = 0; k < k_dim; k++) {
                    data_tmp[k] = i * j_dim * k_dim + j * k_dim + k;
                }
            }
        }
    }

    float *data;
    int64_t i_dim;
    int64_t j_dim;
    int64_t k_dim;
    int64_t i_incr;
    int64_t j_incr;
};

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
        return MatrixCPU(l_x, l_y, l_y);
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

    void ascending() {
        float *data_tmp;
        for (int64_t i = 0; i < row; i++) {
            data_tmp = data + (i * lda);
            for (int64_t j = 0; j < col; j++) {
                data_tmp[j] = i * col + j;
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

inline void add(MatrixCPU3Dim a, MatrixCPU3Dim b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.i_dim; i++) {
        for (int64_t j = 0; j < a.j_dim; j++) {
            a_data = a.data + (i * a.i_incr * a.j_incr) + j * a.j_incr;
            b_data = b.data + (i * b.i_incr * b.j_incr) + j * b.j_incr;
            for (int64_t k = 0; k < a.k_dim; k++) {
                b_data[k] = a_data[k] + 1;
            }
        }
    }
}

inline void divn(MatrixCPU a, float n, MatrixCPU b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);

        cblas_scopy(a.col, a_data, 1, b_data, 1);
        cblas_sscal(a.col, 1 / n, b_data, 1);
    }
}

inline void divn(MatrixCPU4Dim a, float n, MatrixCPU4Dim b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.dims[0]; i++) {
        for (int64_t j = 0; j < a.dims[1]; j++) {
            for (int k = 0; k < a.dims[2]; k++) {
                a_data = a.tensor.data_ptr<float>() + i * a.tensor.stride(0) + j * a.tensor.stride(1) + k * a.tensor.stride(2);
                b_data = b.tensor.data_ptr<float>() + i * b.tensor.stride(0) + j * b.tensor.stride(1) + k * b.tensor.stride(2);
                cblas_scopy(a.dims[3], a_data, 1, b_data, 1);
                cblas_sscal(a.dims[3], 1 / n, b_data, 1);
            }
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

inline void transpose2d(MatrixCPU4Dim a, MatrixCPU4Dim b) {
    for (int i = 0; i < a.dims[0]; i++) {
        for (int j = 0; j < a.dims[1]; j++) {
            MatrixCPU4Dim b_query = b.query(i, j, 0, 0, 1, 1, b.dims[2], b.dims[3]);
            MatrixCPU b_2dim(b_query.tensor.data_ptr<float>(), b.dims[2], b.dims[3], b.tensor.stride(2));
            MatrixCPU4Dim a_query = a.query(i, j, 0, 0, 1, 1, a.dims[2], a.dims[3]);
            MatrixCPU a_2dim(a_query.tensor.data_ptr<float>(), a.dims[2], a.dims[3], a.tensor.stride(2));
            transpose(a_2dim, b_2dim);
        }
    }
}

template <int64_t dim1, int64_t dim2>
inline void transpose4d(MatrixCPU4Dim a, MatrixCPU4Dim b) {
    b.tensor.copy_(a.tensor.transpose(dim1, dim2));
    // for (int i = 0; i < a.dims[0]; i++) {
    //     for (int j = 0; j < a.dims[1]; j++) {
    //         for (int k = 0; k < a.dims[2]; k++) {
    //             for (int l = 0; l < a.dims[3]; l++) {
    //                 std::vector<int> adims{i, j, k, l};
    //                 int bi = 0 == dim1 ? adims[dim2] : (0 == dim2 ? adims[dim1] : adims[0]);
    //                 int bj = 1 == dim1 ? adims[dim2] : (1 == dim2 ? adims[dim1] : adims[1]);
    //                 int bk = 2 == dim1 ? adims[dim2] : (2 == dim2 ? adims[dim1] : adims[2]);
    //                 int bl = 3 == dim1 ? adims[dim2] : (3 == dim2 ? adims[dim1] : adims[3]);
    //                 *b.at(bi, bj, bk, bl) = *a.at(i, j, k, l);
    //             }
    //         }
    //     }
    // }

    // for (int i = 0; i < a.i_dim; i++) {
    //     for (int j = 0; j < a.j_dim; j++) {
    //         MatrixCPU4Dim b_query = b.query(i, j, 0, 0, 1, 1, b.k_dim, b.l_dim);
    //         MatrixCPU b_2dim(b_query.data, b.k_dim, b.l_dim, b.k_incr);
    //         MatrixCPU4Dim a_query = a.query(i, j, 0, 0, 1, 1, a.k_dim, a.l_dim);
    //         MatrixCPU a_2dim(a_query.data, a.k_dim, a.l_dim, a.k_incr);
    //         transpose(a_2dim, b_2dim);
    //     }
    // }
}

inline void softmax(MatrixCPU a, MatrixCPU b) {
    float *a_data;
    float *b_data;

    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);	

        int size = a.col;
        vvexpf(b_data, a_data, &size);

        float exp_sum = 0;

        vDSP_sve(b_data, 1, &exp_sum, a.col);
        cblas_sscal(a.col, 1 / exp_sum, b_data, 1);
    }
}

inline void softmax(MatrixCPU4Dim a, MatrixCPU4Dim b) {
    float *a_data;
    float *b_data;

    for (int i = 0; i < a.dims[0]; i++) {
        for (int j = 0; j < a.dims[1]; j++) {
            for (int k = 0; k < a.dims[2]; k++) {
                a_data = a.tensor.data_ptr<float>() + i * a.tensor.stride(0) + j * a.tensor.stride(1) + k * a.tensor.stride(2);
                b_data = b.tensor.data_ptr<float>() + i * b.tensor.stride(0) + j * b.tensor.stride(1) + k * b.tensor.stride(2);

                int size = a.dims[3];
                vvexpf(b_data, a_data, &size);

                float exp_sum = 0;

                vDSP_sve(b_data, 1, &exp_sum, a.dims[3]);
                cblas_sscal(a.dims[3], 1 / exp_sum, b_data, 1);
            }
        }
    }
}

inline void mmul(MatrixCPU a, MatrixCPU b, MatrixCPU out) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.row, b.col, a.col, 1, a.data, a.lda, b.data, b.lda, 0, out.data, out.lda);
}

inline void mmul2d(MatrixCPU4Dim a, MatrixCPU4Dim b, MatrixCPU4Dim out) {
    float *a_data;
    float *b_data;
    float *out_data;
    for (int i = 0; i < out.dims[0]; i++) {
        for (int j = 0; j < out.dims[1]; j++) {
            a_data = a.tensor.data_ptr<float>() + i * a.tensor.stride(0) + j * a.tensor.stride(1);
            b_data = b.tensor.data_ptr<float>() + i * a.tensor.stride(0) + j * b.tensor.stride(1);
            out_data = out.tensor.data_ptr<float>() + i * out.tensor.stride(0) + j * out.tensor.stride(1); 
            // std::cout << "A" << a.i_dim << " " << a.j_dim << " " << a.k_dim << " " << a.l_dim << " " << a.i_incr << " " << a.j_incr << " " << a.k_incr << std::endl;
            // std::cout << "B" << b.i_dim << " " << b.j_dim << " " << b.k_dim << " " << b.l_dim << " " << b.i_incr << " " << b.j_incr << " " << b.k_incr << std::endl;
            // std::cout << "OUT " << out.i_dim << " " << out.j_dim << " " << out.k_dim << " " << out.l_dim << " " << out.i_incr << " " << out.j_incr << " " << out.k_incr << std::endl;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.dims[2], b.dims[3], a.dims[3], 1, a_data, a.tensor.stride(2), b_data, b.tensor.stride(2), 0, out_data, out.tensor.stride(2));
        }
    }
}

inline void attention(MatrixCPU q, MatrixCPU k, MatrixCPU v, MatrixCPU out) {
    gern::impl::MatrixCPU t = gern::impl::MatrixCPU::allocate(0, 0, k.col, k.row);
    gern::impl::transpose(k, t);
    gern::impl::MatrixCPU matmul = gern::impl::MatrixCPU::allocate(0, 0, q.row, k.row);
    gern::impl::mmul(q, t, matmul);
    gern::impl::MatrixCPU truediv = gern::impl::MatrixCPU::allocate(0, 0, q.row, k.row);
    gern::impl::divn(matmul, sqrt(q.col), truediv);
    gern::impl::MatrixCPU softmax = gern::impl::MatrixCPU::allocate(0, 0, q.row, k.row);
    gern::impl::softmax(truediv, softmax);
    gern::impl::mmul(softmax, v, out);
    matmul.destroy();
    truediv.destroy();
    softmax.destroy();
    t.destroy();
}

inline void attention(MatrixCPU4Dim q, MatrixCPU4Dim k, MatrixCPU4Dim v, MatrixCPU4Dim out) {
    // std::cout << "Q " << q.i_dim << " " << q.j_dim << " " << q.k_dim << " " << q.l_dim << " " << q.i_incr << " " << q.j_incr << " " << q.k_incr << std::endl;
    // std::cout << "K " << k.i_dim << " " << k.j_dim << " " << k.k_dim << " " << k.l_dim << " " << k.i_incr << " " << k.j_incr << " " << k.k_incr << std::endl;
    // std::cout << "V " << v.i_dim << " " << v.j_dim << " " << v.k_dim << " " << v.l_dim << " " << v.i_incr << " " << v.j_incr << " " << v.k_incr << std::endl;
    // std::cout << "OUT " << out.i_dim << " " << out.j_dim << " " << out.k_dim << " " << out.l_dim << " " << out.i_incr << " " << out.j_incr << " " << out.k_incr << std::endl;
    MatrixCPU4Dim kt(k.dims[0], k.dims[1], k.dims[3], k.dims[2]);
    transpose2d(k, kt);
    MatrixCPU4Dim q_kt(q.dims[0], q.dims[1], q.dims[2], k.dims[2]);
    mmul2d(q, kt, q_kt);
    float sqrt_dk = sqrt(q.dims[3]);
    MatrixCPU4Dim sm_in(q.dims[0], q.dims[1], q.dims[2], k.dims[2]); 
    divn(q_kt, sqrt_dk, sm_in);
    MatrixCPU4Dim sm_out(q.dims[0], q.dims[1], q.dims[2], k.dims[2]); 
    softmax(sm_in, sm_out);
    mmul2d(sm_out, v, out);
    // MatrixCPU kt(k.col, k.row, k.row);
    // transpose(k, kt);
    // MatrixCPU q_kt(q.row, k.row, k.row);
    // mmul(q, kt, q_kt);
    // float sqrt_dk = sqrt(q.col);
    // MatrixCPU sm_in(q.row, k.row, k.row);
    // divn(q_kt, sqrt_dk, sm_in);
    // MatrixCPU sm_out(q.row, k.row, k.row);
    // softmax(sm_in, sm_out);
    // mmul(sm_out, v, out);
}


inline void exp_matrix(MatrixCPU a, MatrixCPU b) {
    float *a_data;
    float *b_data;
    for (int64_t i = 0; i < a.row; i++) {
        a_data = a.data + (i * a.lda);
        b_data = b.data + (i * b.lda);
        int size = a.col;
        vvexpf(b_data, a_data, &size);
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

inline void matrix_multiply(MatrixCPU a, MatrixCPU b, MatrixCPU c, int64_t k_dummy) {
    float *a_data;
    float *b_data;
    float *c_data;
    for (int64_t i = 0; i < c.row; i++) {
        c_data = c.data + (i * c.lda);
        a_data = a.data + (i * a.lda);
        for (int64_t j = 0; j < c.col; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < k_dummy; k++) {
                b_data = b.data + (k * b.lda) + j;
                sum += a_data[k] * b_data[0];
            }
            c_data[j] += sum;
        }
    }
}

}  // namespace impl
}  // namespace gern