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
            Tile(outputDS["size"], step)(
                add_1(inputDS, outputDS)),
            {{Grid::Dim::BLOCK_DIM_X, step}});
}