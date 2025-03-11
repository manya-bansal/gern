#include "cassert"
#include "cpu-matrix.h"

template <int64_t tc, int64_t tr>
void function_15(library::impl::MatrixCPU &A, library::impl::MatrixCPU &B,
                 library::impl::MatrixCPU &C, int64_t k_dim) {
  for (int64_t _gern_j_2_6_11 = 0; (_gern_j_2_6_11 < C.col);
       _gern_j_2_6_11 = (_gern_j_2_6_11 + tc)) {

    for (int64_t _gern_i_1_5 = 0; (_gern_i_1_5 < C.row);
         _gern_i_1_5 = (_gern_i_1_5 + tr)) {

      int64_t _gern_j_2 = _gern_j_2_6_11;
      constexpr int64_t _gern_tj_4 = tc;
      int64_t _gern_i_1 = _gern_i_1_5;
      constexpr int64_t _gern_ti_3 = tr;

      auto _query_C_16 = C.query(_gern_i_1, _gern_j_2, _gern_ti_3, _gern_tj_4);

      auto _query_A_17 = A.query(_gern_i_1, 0, _gern_ti_3, k_dim);

      auto _query_B_18 = B.query(0, _gern_j_2, k_dim, _gern_tj_4);

      library::impl::matrix_multiply(_query_A_17, _query_B_18, _query_C_16,
                                     k_dim);
    }
  }
}

extern "C" {
void hook_function_15(void **args) {
  constexpr int64_t tc = 5;
  constexpr int64_t tr = 5;
  library::impl::MatrixCPU &A = *((library::impl::MatrixCPU *)args[0]);
  library::impl::MatrixCPU &B = *((library::impl::MatrixCPU *)args[1]);
  library::impl::MatrixCPU &C = *((library::impl::MatrixCPU *)args[2]);
  int64_t k_dim = *((int64_t *)args[3]);
  function_15<tc, tr>(A, B, C, k_dim);
}
}
