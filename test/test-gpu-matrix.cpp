#include "annotations/visitor.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/gpu-matrix.h"
#include "library/matrix/impl/gpu-matrix.h"
#include "test-gpu-utils.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringGPU, MatrixGPUAddNoBind) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixGPU("output_con"));

    annot::MatrixAddGPU add;
    Variable l_x("l_x");
    Variable l_y("l_y");

    Composable program =
        Global(
            Tile(outputDS["row"], l_x)(
                Tile(outputDS["col"], l_y)(
                    add(inputDS, outputDS))));

    Runner run(program);

    run.compile(test::gpuRunner(std::vector<std::string>{"matrix", "array"}));

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixGPU b(row_val, col_val, row_val);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    impl::MatrixCPU result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 3.0f);
    }

    // Run with a couple more settings.
    a.vvals(7.0f);
    l_y_val = 2;
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));
    result = b.get();
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 8.0f);
    }

    a.destroy();
    b.destroy();
    result.destroy();
}

TEST(LoweringGPU, MatrixGPUAddSingleBind) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixGPU("output_con"));

    annot::MatrixAddGPU add;
    Variable l_x("l_x");
    Variable l_y("l_y");

    Composable program = Global(
        (Tile(outputDS["row"], l_x) || Grid::Unit::BLOCK_X)(
            Tile(outputDS["col"], l_y)(
                add(inputDS, outputDS))));

    Runner run(program);

    run.compile(test::gpuRunner(std::vector<std::string>{"matrix", "array"}));

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixGPU b(row_val, col_val, row_val);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    impl::MatrixCPU result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 3.0f);
    }

    // Run with a couple more settings.
    a.vvals(7.0f);
    l_y_val = 2;
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));
    result = b.get();
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 8.0f);
    }

    a.destroy();
    b.destroy();
    result.destroy();
}

TEST(LoweringGPU, MatrixGPUAddDoubleBind) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixGPU("output_con"));

    annot::MatrixAddGPU add;
    Variable l_x("l_x");
    Variable l_y("l_y");

    Composable program =
        Global((Tile(outputDS["row"], l_x) || Grid::Unit::BLOCK_X)(
            (Tile(outputDS["col"], l_y) || Grid::Unit::BLOCK_Y)(
                add(inputDS, outputDS))));

    Runner run(program);

    run.compile(test::gpuRunner(std::vector<std::string>{"matrix", "array"}));

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixGPU b(row_val, col_val, row_val);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    impl::MatrixCPU result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 3.0f);
    }

    // Run with a couple more settings.
    a.vvals(7.0f);
    l_y_val = 2;
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));
    result = b.get();
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 8.0f);
    }

    a.destroy();
    b.destroy();
    result.destroy();
}