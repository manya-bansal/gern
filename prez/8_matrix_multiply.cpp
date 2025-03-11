#include "helpers.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "our-matrix-multiply.h"
using namespace gern;

int main() {
    // ***** PROGRAM DEFINITION *****
    auto A = mk_matrix("A");
    auto B = mk_matrix("B");
    auto C = mk_matrix("C");

    gern::annot::OurMatrixMultiply matrix_multiply;
    Variable k_dim("k_dim");

    Variable tr("tr");
    Variable tc("tc");
    Variable tk("tk");

    Composable program({
        Tile(C["col"], tc.bind(5))(
            Tile(C["row"], tr.bind(5))(
                matrix_multiply(A, B, C, k_dim))),

    });

    // ***** PROGRAM EVALUATION *****
    library::impl::MatrixCPU a(10, 10);
    a.ascending();
    library::impl::MatrixCPU b(10, 10);
    b.ascending();
    library::impl::MatrixCPU c(10, 10);
    b.vvals(0.0f);
    int64_t k_dim_val = 10;

    auto runner = compile_program(program);
    runner.evaluate({
        {"A", &a},
        {"B", &b},
        {"C", &c},
        {"k_dim", &k_dim_val},
    });

    // ***** SANITY CHECK *****
    for (int64_t i = 0; i < 10; i++) {
        for (int64_t j = 0; j < 10; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < 10; k++) {
                sum += a(i, k) * b(k, j);
            }
            assert(c(i, j) == sum);
        }
    }
}