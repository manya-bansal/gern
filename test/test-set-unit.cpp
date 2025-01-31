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

TEST(SetGrid, NoMap) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("output_con"));

    annot::Add1GPUTemplate add_1;
    Variable step("step");
    Variable blk("blk");

    Composable program =
        add_1(tempDS, outputDS);

    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::SCALAR);

    program =
        (Tile(outputDS["size"], step))(
            add_1(tempDS, outputDS));
    // No change still.
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::SCALAR);

    program =
        (Tile(outputDS["size"], blk))(
            (Tile(outputDS["size"], step))(
                add_1(tempDS, outputDS)));
    // No change still.
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::SCALAR);

    // Try a multi stage pipeline.
    program =
        (Tile(outputDS["size"], blk))(
            (Tile(outputDS["size"], step))(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS)));
    // No change still.
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::SCALAR);
}

TEST(SetGrid, WithMap) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("output_con"));

    annot::Add1GPUTemplate add_1;
    Variable step("step");
    Variable blk("blk");

    Composable program =
        (Tile(outputDS["size"], step) || Grid::Unit::THREAD_X)(
            add_1(tempDS, outputDS));
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::THREADS);

    program =
        (Tile(outputDS["size"], step) || Grid::Unit::BLOCK_X)(
            add_1(tempDS, outputDS));
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::BLOCK);

    program =
        (Tile(outputDS["size"], blk) || Grid::Unit::BLOCK_X)(
            (Tile(outputDS["size"], step) || Grid::Unit::THREAD_X)(
                add_1(tempDS, outputDS)));
    // No change still.
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::BLOCK);

    program =
        Composable({(Tile(tempDS["size"], blk) || Grid::Unit::BLOCK_X)(
                        add_1(inputDS, tempDS)),
                    (Tile(outputDS["size"], step) || Grid::Unit::THREAD_X)(
                        add_1(tempDS, outputDS))});
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::BLOCK);

    // Try with no child tiling.
    program =
        Composable({(Tile(tempDS["size"], blk) || Grid::Unit::BLOCK_X)(
                        add_1(inputDS, tempDS)),
                    (Tile(outputDS["size"], step))(
                        add_1(tempDS, outputDS))});
    ASSERT_TRUE(getLevel(program.getAnnotation().getOccupiedUnits()) == Grid::Level::BLOCK);
}