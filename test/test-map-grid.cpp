#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/gpu-array.h"
#include "test-gpu-utils.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(MapToGrid, RespectHierarchy) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("output_con"));

    annot::Add1GPUTemplate add_1;
    Variable step("step");
    Variable blk("blk");

    // Not possible to tile the outer loop to a part of the grid that
    // is lower on the hierarchy than its inner loop.
    ASSERT_THROW(
        (Tile(outputDS["size"], blk) || Grid::Unit::THREAD_X)(
            (Tile(outputDS["size"], step) || Grid::Unit::BLOCK_X)(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS))),
        error::UserError);

    // Not possible to map the outer loop to a part of the grid that
    // is lower on the hierarchy than its inner loop. Catch even with
    // an unmapped loop in the middle.
    ASSERT_THROW(
        (Tile(outputDS["size"], blk) || Grid::Unit::THREAD_X)(
            Tile(outputDS["size"], blk)(
                (Tile(outputDS["size"], step) || Grid::Unit::BLOCK_X)(
                    add_1(inputDS, tempDS),
                    add_1(tempDS, outputDS)))),
        error::UserError);

    // This should be ok.
    ASSERT_NO_THROW(
        (Tile(outputDS["size"], blk) || Grid::Unit::THREAD_X)(
            Tile(outputDS["size"], blk)(
                (Tile(outputDS["size"], step) || Grid::Unit::THREAD_Y)(
                    add_1(inputDS, tempDS),
                    add_1(tempDS, outputDS)))));

    // This is ok.
    ASSERT_NO_THROW(
        Composable({
            (Tile(tempDS["size"], blk) || Grid::Unit::BLOCK_X)(
                add_1(inputDS, tempDS)),
            (Tile(outputDS["size"], step) || Grid::Unit::THREAD_X)(
                add_1(tempDS, outputDS)),
        }));
}