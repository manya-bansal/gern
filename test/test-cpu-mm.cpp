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
    Variable l_x("l_x");
    Variable l_x_2("l_x_2");
    Variable l_y("l_y");
    Variable l_y_2("l_y_2");
    Variable k("k");
    Variable k_2("k_2");

    Composable program = {
        (Tile(C_DS["row"], l_x))(
            Tile(C_DS["col"], l_y)(
                (Tile(C_DS["row"], l_x_2))(
                    Tile(C_DS["col"], l_y_2)(
                        (Reduce(A_DS["col"], k.bind(5)))(
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

    int64_t l_x_val = 5;
    int64_t l_x_2_val = 1;
    int64_t l_y_val = 5;
    int64_t l_y_2_val = 1;

    run.evaluate({
        {A_DS.getName(), &a},
        {B_DS.getName(), &b},
        {C_DS.getName(), &c},
        {l_x.getName(), &l_x_val},
        {l_x_2.getName(), &l_x_2_val},
        {l_y.getName(), &l_y_val},
        {l_y_2.getName(), &l_y_2_val},
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
