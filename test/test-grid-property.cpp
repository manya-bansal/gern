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

    auto inputDS = std::make_shared<const annot::ArrayGPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayGPU>("output_con");

    annot::addGPU add_f;
    Variable v("v");

    ASSERT_THROW(
        (add_f[{{"end", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]),
        error::UserError);
    ASSERT_NO_THROW(
        (add_f[{{"end", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]));
}

TEST(SetGridProperty, BindIntervalVar) {

    auto inputDS = std::make_shared<const annot::ArrayGPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayGPU>("output_con");

    annot::addGPU add_f;
    Variable v("v");

    ASSERT_NO_THROW((add_f[{{"x", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]));
    ASSERT_THROW(
        (add_f[{{"x", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]),
        error::UserError);
}

TEST(SetGridProperty, BindNestedIntervalVar) {

    auto inputDS = std::make_shared<const annot::ArrayGPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayGPU>("output_con");

    annot::reductionGPU reduce_f;
    Variable v("v");

    ASSERT_NO_THROW((reduce_f[{{"r", v.bindToGrid(Grid::Property::BLOCK_ID_X)}}]));
    ASSERT_THROW(
        (reduce_f[{{"r", v.bindToGrid(Grid::Property::BLOCK_DIM_X)}}]),
        error::UserError);
}

TEST(SetGridProperty, DoubleBind) {

    auto inputDS = std::make_shared<const annot::ArrayGPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayGPU>("output_con");

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
