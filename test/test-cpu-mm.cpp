#include "compose/composable.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringCPU, MatrixMultiply) {
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixCPU("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixCPU("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixCPU("C"));

    annot::MatrixMultiplyCPU matrix_multiply;
    Variable ti("ti");
    Variable ti_2("ti_2");
    Variable tj("tj");
    Variable tj_2("tj_2");
    Variable k("k");
    Variable k_2("k_2");

    Composable program = {
        (Tile(C_DS["row"], ti))(
            Tile(C_DS["col"], tj)(
                (Reduce(A_DS["col"], k.bind(5)))(
                    (Tile(C_DS["row"], ti_2))(
                        Tile(C_DS["col"], tj_2)(
                            Reduce(A_DS["col"], k_2.bind(1))(
                                matrix_multiply(A_DS, B_DS, C_DS)))))))};

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t num_row = 10;
    int64_t num_col = 10;

    impl::MatrixCPU a(num_row, num_col, num_col);
    a.ascending();
    impl::MatrixCPU b(num_row, num_col, num_col);
    b.ascending();

    impl::MatrixCPU c(num_row, num_col, num_col);
    c.vvals(0.0f);

    int64_t ti_val = 5;
    int64_t ti_2_val = 1;
    int64_t tj_val = 5;
    int64_t tj_2_val = 1;

    run.evaluate({
        {A_DS.getName(), &a},
        {B_DS.getName(), &b},
        {C_DS.getName(), &c},
        {ti.getName(), &ti_val},
        {ti_2.getName(), &ti_2_val},
        {tj.getName(), &tj_val},
        {tj_2.getName(), &tj_2_val},
    });

    impl::MatrixCPU ref_c(num_row, num_col, num_col);
    ref_c.vvals(0.0f);
    impl::matrix_multiply(a, b, ref_c);

    for (int i = 0; i < num_row * num_col; i++) {
        ASSERT_TRUE(c.data[i] == ref_c.data[i]);
    }

    a.destroy();
    b.destroy();
    c.destroy();
    ref_c.destroy();
}
