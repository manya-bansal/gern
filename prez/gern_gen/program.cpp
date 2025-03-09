#tcinclude "cassert"
#include "cpu-matrix.h"

template <int64_t tc, int64_t tk, int64_t tr>
void function_25(gern::impl::MatrixCPU &A, gern::impl::MatrixCPU &B,
                 gern::impl::MatrixCPU &C, int64_t k_dim) {
  for (int64_t _gern_j_2_8_14_20 = 0; (_gern_j_2_8_14_20 < C.col);
       _gern_j_2_8_14_20 = (_gern_j_2_8_14_20 + tc)) {

    for (int64_t _gern_i_1_7_13 = 0; (_gern_i_1_7_13 < C.row);
         _gern_i_1_7_13 = (_gern_i_1_7_13 + tr)) {

      int64_t _gern_j_2_8 = _gern_j_2_8_14_20;
      constexpr int64_t _gern_tj_5_11 = tc;
      int64_t _gern_i_1_7 = _gern_i_1_7_13;
      constexpr int64_t _gern_ti_4_10 = tr;

      auto _query_C_26 =
          C.query(_gern_i_1_7, _gern_j_2_8, _gern_ti_4_10, _gern_tj_5_11);

      for (int64_t _gern_k_3_9 = 0; (_gern_k_3_9 < k_dim);
           _gern_k_3_9 = (_gern_k_3_9 + tk)) {

        int64_t _gern_j_2 = _gern_j_2_8_14_20;
        constexpr int64_t _gern_tj_5 = tc;
        int64_t _gern_i_1 = _gern_i_1_7_13;
        constexpr int64_t _gern_ti_4 = tr;

        int64_t _gern_k_3 = _gern_k_3_9;
        constexpr int64_t _gern_tk_6 = tk;

        auto _query_A_27 =
            A.query(_gern_i_1, _gern_k_3, _gern_ti_4, _gern_tk_6);

        auto _query_B_28 =
            B.query(_gern_k_3, _gern_j_2, _gern_tk_6, _gern_tj_5);

        gern::impl::matrix_multiply(_query_A_27, _query_B_28, _query_C_26, tk);
      }
    }
  }
}

extern "C" {
void hook_function_25(void **args) {
  constexpr int64_t tc = 5;
  constexpr int64_t tk = 1;
  constexpr int64_t tr = 5;
  gern::impl::MatrixCPU &A = *((gern::impl::MatrixCPU *)args[0]);
  gern::impl::MatrixCPU &B = *((gern::impl::MatrixCPU *)args[1]);
  gern::impl::MatrixCPU &C = *((gern::impl::MatrixCPU *)args[2]);
  int64_t k_dim = *((int64_t *)args[3]);
  function_25<tc, tk, tr>(A, B, C, k_dim);
}
}
