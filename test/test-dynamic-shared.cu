#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/gpu-array.h"
#include "library/array/impl/gpu-array.h"
#include "library/matrix/annot/gpu-matrix.h"
#include "library/matrix/impl/gpu-matrix.h"
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

TEST(LoweringGPU, StageIntoShared) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixGPUStageSmem("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixGPUStageSmem("output_con"));

    annot::MatrixAddGPUBlocks add_blocks;
    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable smem_size("smem_size");
    Variable thread_x("thread_x");
    Variable thread_y("thread_y");

    Composable program =
        Global((Tile(outputDS["row"], l_x) || Grid::Unit::BLOCK_X)(
                   (Tile(outputDS["col"], l_y) || Grid::Unit::BLOCK_Y)(
                       add_blocks(inputDS, outputDS))),
               {{Grid::Dim::BLOCK_DIM_X, thread_x}, {Grid::Dim::BLOCK_DIM_Y, thread_y}},
               smem_size, test::TrivialManager(smem_size));

    Runner run(program);

    run.compile(test::gpuRunner(std::vector<std::string>{"matrix", "array"}));

    int64_t row_val = 4;
    int64_t col_val = 4;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.ascending();
    impl::MatrixGPU b(row_val, col_val, row_val);

    int64_t l_x_val = 4;
    int64_t l_y_val = 4;
    int64_t thread_x_val = 2;
    int64_t thread_y_val = 2;
    int64_t smem_size_val = 1024;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
        {smem_size.getName(), &smem_size_val},
        {thread_x.getName(), &thread_x_val},
        {thread_y.getName(), &thread_y_val},
    }));

    impl::MatrixCPU result = b.get();
    impl::MatrixCPU a_host = a.get();
    std::cout << "Result: " << result << std::endl;
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == a_host.data[i] + 1);
    }

    a.destroy();
    b.destroy();
    result.destroy();
    a_host.destroy();
}
