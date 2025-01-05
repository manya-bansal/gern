#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "functions/elementwise.h"
#include "functions/reduction.h"
#include "library/array/gpu-array-lib.h"
#include "utils/error.h"

#include "config.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(SetGridProperty, BindStableVar) {

    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::addGPU add_f;
    Variable v("v");

    ASSERT_THROW(
        (add_f[{{"end", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]),
        error::UserError);
    ASSERT_NO_THROW(
        (add_f[{{"end", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]));
}

TEST(SetGridProperty, BindIntervalVar) {

    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::addGPU add_f;
    Variable v("v");

    ASSERT_NO_THROW((add_f[{{"x", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]));
    ASSERT_THROW(
        (add_f[{{"x", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]),
        error::UserError);
}

TEST(SetGridProperty, BindNestedIntervalVar) {

    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::reduction reduce_f;
    Variable v("v");

    ASSERT_NO_THROW((reduce_f[{{"r", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]));
    ASSERT_THROW(
        (reduce_f[{{"r", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]),
        error::UserError);
}

TEST(SetGridProperty, DoubleBind) {

    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::reduction reduce_f;
    Variable v("v");

    ASSERT_NO_THROW((reduce_f[{
        {"r", v.bindToGrid(Grid::Property::BLOCK_ID_X)},
        {"x", v.bindToGrid(Grid::Property::BLOCK_ID_Y)},
    }]));
    ASSERT_THROW(
        (reduce_f[{
            {"r", v.bindToGrid(Grid::Property::BLOCK_ID_X)},
            {"x", v.bindToGrid(Grid::Property::BLOCK_DIM_Y)},
        }]),
        error::UserError);
}
