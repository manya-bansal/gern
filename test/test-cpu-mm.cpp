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

    Composable program = {
        (Tile(C_DS["row"], l_x))(
            Tile(C_DS["col"], l_y)(
                (Tile(C_DS["row"], l_x_2))(
                    matrix_multiply(A_DS, B_DS, C_DS))))};

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));
}