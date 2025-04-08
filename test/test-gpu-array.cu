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

    annot::Add1GPU add_1;
    Variable v("v");
    Variable step("step");

    // No interval variable is being mapped to the grid,
    // this implementation runs the entire computation on a
    // single thread.
    Composable program =
        Global(Tile(outputDS["size"], step)(
            add_1(inputDS, outputDS)));

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

    annot::Add1GPU add_1;
    Variable v("v");
    Variable step("step");
    Variable blk("blk");

    Composable program =
        Global((Tile(outputDS["size"], step) || Grid::Unit::BLOCK_X)(
            add_1(inputDS, outputDS)));

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

    annot::Add1GPU add_1;
    Variable v("v");
    Variable step("step");
    Variable blk("blk");

    Composable program =
        Global(
            (Tile(outputDS["size"], blk) || Grid::Unit::BLOCK_X)(
                (Tile(outputDS["size"], step) || Grid::Unit::THREAD_X)(
                    add_1(inputDS, outputDS))));

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

    annot::Add1GPUTemplate add_1;
    Variable x("x");
    Variable end("end");
    Variable step("step");
    Variable blk("blk");

    Composable program =
        Global((Tile(outputDS["size"], blk) || Grid::Unit::BLOCK_X)(
            (Tile(outputDS["size"], step.bind(1)) || Grid::Unit::THREAD_X)(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS))));

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

TEST(LoweringGPU, RepeatUnit) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayStaticGPU("output_con"));

    annot::Add1GPU add_1;
    Variable x("x");
    Variable end("end");
    Variable step("step");
    Variable step2("step2");
    Variable step3("step3");
    Variable blk("blk");

    Composable program =
        Global((Tile(outputDS["size"], blk.bind(8)) || Grid::Unit::BLOCK_X)(
            (Tile(outputDS["size"], step.bind(4)) || Grid::Unit::BLOCK_X)(
                (Tile(outputDS["size"], step2.bind(2)) || Grid::Unit::THREAD_X)(
                    (Tile(outputDS["size"], step3.bind(1)) || Grid::Unit::THREAD_Y)(
                        (Tile(outputDS["size"], step3.bind(1)) || Grid::Unit::THREAD_X)(
                            add_1(inputDS, tempDS),
                            add_1(tempDS, outputDS)))))));

    Runner run(program);
    run.compile(test::gpuRunner("array"));

    impl::ArrayGPU a(16);
    a.ascending();
    impl::ArrayGPU b(16);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    }));

    impl::ArrayCPU result = b.get();
    impl::ArrayCPU a_host = a.get();

    std::cout << "Result: " << result << std::endl;

    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(result.data[i] == (a_host.data[i] + 2));
    }

    a.destroy();
    b.destroy();
    result.destroy();
    a_host.destroy();
}
