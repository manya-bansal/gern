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

TEST(LoweringGPU, SingleElemFunctionNoBind) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::add_1_GPU add_f;
    Variable v("v");
    Variable step("step");

    // No interval variable is being mapped to the grid,
    // this implementation runs the entire computation on a
    // single thread.
    Composable program =
        Tile(outputDS["size"], step)(
            add_f(inputDS, outputDS));

    program.callAtDevice();
    Runner run(program);

    run.compile(test::gpuRunner("array"));

    // Now, actually run the function.
    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t step_val = 10;

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

    // Try running with insufficient number
    ASSERT_THROW(run.evaluate({
                     {inputDS.getName(), &a},
                     {outputDS.getName(), &b},
                 }),
                 error::UserError);

    auto dummyDS = AbstractDataTypePtr(new const annot::ArrayGPU("dummy_ds"));
    // Try running the correct number of arguments,
    // but with the wrong reference data-structure.
    ASSERT_THROW(run.evaluate({
                     {inputDS.getName(), &a},
                     {dummyDS.getName(), &b},
                 }),
                 error::UserError);

    a.destroy();
    b.destroy();
    result.destroy();
    a_host.destroy();
}

TEST(LoweringGPU, SingleElemFunctionBind) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::add_1_GPU add_1;
    Variable v("v");
    Variable step("step");
    Variable blk("blk");

    Composable program =
        (Tile(outputDS["size"], step) || Grid::Property::BLOCK_ID_X)(
            add_1(inputDS, outputDS));

    program.callAtDevice();
    Runner run(program);
    run.compile(test::gpuRunner("array"));
    // Now, actually run the function.
    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t step_val = 1;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {step.getName(), &step_val},
    }));

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

TEST(LoweringGPU, DoubleBind) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::add_1_GPU add_f;
    Variable v("v");
    Variable step("step");
    Variable blk("blk");

    Composable program =
        (Tile(outputDS["size"], blk) || Grid::Property::BLOCK_ID_X)(
            (Tile(outputDS["size"], step) || Grid::Property::THREAD_ID_X)(
                add_f(inputDS, outputDS)));

    program.callAtDevice();
    Runner run(program);
    run.compile(test::gpuRunner("array"));
    // Now, actually run the function.
    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t step_val = 1;
    int64_t blk_val = 1;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {step.getName(), &step_val},
        {blk.getName(), &blk_val},
    }));

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

TEST(LoweringGPU, MultiArray) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("output_con"));

    annot::add_1_GPU_Template add_1;
    Variable x("x");
    Variable end("end");
    Variable step("step");
    Variable blk("blk");

    Composable program =
        (Tile(outputDS["size"], blk) || Grid::Property::BLOCK_ID_X)(
            (Tile(outputDS["size"], step.bindToInt64(1)) || Grid::Property::THREAD_ID_X)(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS)));

    program.callAtDevice();
    Runner run(program);
    run.compile(test::gpuRunner("array"));

    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t blk_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {blk.getName(), &blk_val},
    }));

    impl::ArrayCPU result = b.get();
    impl::ArrayCPU a_host = a.get();
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(result.data[i] == (a_host.data[i] + 2));
    }

    a.destroy();
    b.destroy();
    result.destroy();
    a_host.destroy();
}