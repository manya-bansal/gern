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

TEST(SharedMemoryManager, SetSize) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::Add1GPU add_1;
    Variable v("v");
    Variable step("step");
    Variable smem_size("smem_size");

    // No interval variable is being mapped to the grid,
    // this implementation runs the entire computation on a
    // single thread.
    Composable program =
        Global(Tile(outputDS["size"], step)(
                   add_1(inputDS, outputDS)),
               {}, smem_size);

    Runner run(program);

    run.compile(test::gpuRunner("array"));

    // Now, actually run the function.
    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t step_val = 10;
    int64_t smem_size_val = 1024;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {step.getName(), &step_val},
        {smem_size.getName(), &smem_size_val},
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

TEST(LoweringGPU, WithSetup) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayGPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayGPU("output_con"));

    annot::Add1GPU add_1;
    Variable v("v");
    Variable step("step");
    Variable smem_size("smem_size");

    // No interval variable is being mapped to the grid,
    // this implementation runs the entire computation on a
    // single thread.
    Composable program =
        Global(Tile(outputDS["size"], step)(
                   add_1(inputDS, outputDS)),
               {}, smem_size, test::TrivialManager(smem_size));

    Runner run(program);

    run.compile(test::gpuRunner("array"));

    // Now, actually run the function.
    impl::ArrayGPU a(10);
    a.ascending();
    impl::ArrayGPU b(10);
    int64_t step_val = 10;
    int64_t smem_size_val = 1024;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {step.getName(), &step_val},
        {smem_size.getName(), &smem_size_val},
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
