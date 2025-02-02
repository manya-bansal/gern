#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/gpu-array.h"
#include "library/array/impl/gpu-array.h"
#include "test-gpu-utils.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(SetGrid, SimpleSet) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::AddArrayThreads add_1;
    Variable v("v");
    Variable step("step");

    // No interval variable is being mapped to the grid,
    // this implementation runs the entire computation on a
    // single thread.
    ASSERT_NO_THROW(
        Global(
            Tile(outputDS["size"], step)(
                add_1(inputDS, outputDS))));

    ASSERT_NO_THROW(
        Global(
            Tile(outputDS["size"], step)(
                add_1(inputDS, outputDS)),
            {{Grid::Dim::BLOCK_DIM_X, step}}));

    // Do not allow to set the launch dimension of block y.
    ASSERT_THROW(
        Global(
            Tile(outputDS["size"], step)(
                add_1(inputDS, outputDS)),
            {{Grid::Dim::BLOCK_DIM_Y, step}}),
        error::UserError);
}

// Try running a program that requires the grid to be set.
TEST(SetGrid, RunWithParam) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::AddArrayThreads add_1;
    Variable v("v");
    Variable step("step");

    Composable program =
        Global(
            Tile(outputDS["size"], step)(  // This tile is useless,
                                           // though should not result in incorrect code.
                add_1(inputDS, outputDS)),
            {{Grid::Dim::BLOCK_DIM_X, step}});

    Runner run(program);
    run.compile(test::gpuRunner("array"));

    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t step_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {step.getName(), &step_val},
    }));

    // impl::ArrayCPU result = b.get();
    // Make sure we got the correct answer.
    impl::ArrayCPU result = b.get();
    impl::ArrayCPU a_host = a.get();

    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(result.data[i] == (a_host.data[i] + 1));
    }

    a.destroy();
    b.destroy();
    result.destroy();
    a_host.destroy();
}

TEST(SetGrid, CatchStatic) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::AddArrayThreads add_1;
    Variable v("v");
    Variable step("step");
    Variable block_x_dim("block_x_dim");

    Composable program =
        Global(
            Tile(outputDS["size"], step.bindToInt64(2))(  // This tile is useless,
                                                          // though should not result in incorrect code.
                add_1(inputDS, outputDS)),
            {{Grid::Dim::BLOCK_DIM_X, block_x_dim.bindToInt64(5)}});

    Runner run(program);
    run.compile(test::gpuRunner("array"));
    ASSERT_TRUE(false);
}