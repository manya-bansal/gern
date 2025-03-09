#include "helpers.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"

int main() {
    // ***** PROGRAM DEFINITION *****
    auto input = mk_matrix("input");
    auto output = mk_matrix("output");

    annot::MatrixAddCPU add_1;

    Composable program({
        add_1(input, output),
    });

    // ***** PROGRAM EVALUATION *****
    gern::impl::MatrixCPU a(10, 10, 10);
    a.ascending();
    gern::impl::MatrixCPU b(10, 10, 10);

    auto runner = compile_program(program);
    runner.evaluate({
        {"input", &a},
        {"output", &b},
    });

    // ***** SANITY CHECK *****
    for (int i = 0; i < a.col; i++) {
        for (int j = 0; j < a.row; j++) {
            assert(a(i, j) + 1 == b(i, j));
        }
    }
}
