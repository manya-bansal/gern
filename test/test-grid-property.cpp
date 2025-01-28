#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "config.h"
#include "library/array/annot/gpu-array.h"
#include "test-utils.h"
#include "utils/error.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(SetGridProperty, BindStableVar) {
    annot::addGPU add_1;
    Variable v("v");

    ASSERT_THROW(
        (add_1[{{"end", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]),
        error::UserError);
    ASSERT_NO_THROW(
        (add_1[{{"end", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]));
}

TEST(SetGridProperty, BindIntervalVar) {
    annot::addGPU add_1;
    Variable v("v");

    ASSERT_NO_THROW((add_1[{{"x", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]));
    ASSERT_THROW(
        (add_1[{{"x", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]),
        error::UserError);
}

TEST(SetGridProperty, BindNestedIntervalVar) {
    annot::reductionGPU reduce_f;
    Variable v("v");

    ASSERT_NO_THROW((reduce_f[{{"r", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]));
    ASSERT_THROW(
        (reduce_f[{{"r", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]),
        error::UserError);
}

TEST(SetGridProperty, DoubleBind) {
    annot::reductionGPU reduce_f;
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
