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

    annot::add_1_GPU_Template add_1;
    Variable step("step");
    Variable blk("blk");

    Composable program =
        add_1(tempDS, outputDS);

    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::SCALAR);

    program =
        (Tile(outputDS["size"], step))(
            add_1(tempDS, outputDS));
    // No change still.
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::SCALAR);

    program =
        (Tile(outputDS["size"], blk))(
            (Tile(outputDS["size"], step))(
                add_1(tempDS, outputDS)));
    // No change still.
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::SCALAR);

    // Try a multi stage pipeline.
    program =
        (Tile(outputDS["size"], blk))(
            (Tile(outputDS["size"], step))(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS)));
    // No change still.
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::SCALAR);
}

TEST(SetGrid, WithMap) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("output_con"));

    annot::add_1_GPU_Template add_1;
    Variable step("step");
    Variable blk("blk");

    Composable program =
        (Tile(outputDS["size"], step) || Grid::Property::THREAD_ID_X)(
            add_1(tempDS, outputDS));
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::THREADS);

    program =
        (Tile(outputDS["size"], step) || Grid::Property::BLOCK_ID_X)(
            add_1(tempDS, outputDS));
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::BLOCK);

    program =
        (Tile(outputDS["size"], blk) || Grid::Property::BLOCK_ID_X)(
            (Tile(outputDS["size"], step) || Grid::Property::THREAD_ID_X)(
                add_1(tempDS, outputDS)));
    // No change still.
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::BLOCK);

    program =
        Composable({(Tile(tempDS["size"], blk) || Grid::Property::BLOCK_ID_X)(
                        add_1(inputDS, tempDS)),
                    (Tile(outputDS["size"], step) || Grid::Property::THREAD_ID_X)(
                        add_1(tempDS, outputDS))});
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::BLOCK);

    // Try with no child tiling.
    program =
        Composable({(Tile(tempDS["size"], blk) || Grid::Property::BLOCK_ID_X)(
                        add_1(inputDS, tempDS)),
                    (Tile(outputDS["size"], step))(
                        add_1(tempDS, outputDS))});
    ASSERT_TRUE(program.getAnnotation().getOccupiedUnit() == Grid::Unit::BLOCK);
}