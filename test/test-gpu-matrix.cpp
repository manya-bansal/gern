#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/gpu-matrix.h"
#include "library/matrix/impl/gpu-matrix.h"
#include "test-gpu-utils.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringGPU, MatrixGPUAddNoBind) {
    auto inputDS = new const annot::MatrixGPU("input_con");
    auto outputDS = new const annot::MatrixGPU("output_con");

    annot::MatrixAddGPU add;
    Variable row("row");
    Variable col("col");
    Variable l_x("l_x");
    Variable l_y("l_y");

    std::vector<Compose> c = {add[{
        {"row", row},
        {"col", col},
        {"l_x", l_x},
        {"l_y", l_y},
    }](inputDS, outputDS)};

    Pipeline p(c);
    p.at_device();
    Runner run(p);

    run.compile(test::gpuRunner("matrix"));

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixGPU b(row_val, col_val, row_val);
    b.vvals(3.0f);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {row.getName(), &row_val},
        {col.getName(), &col_val},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    impl::MatrixCPU result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 5.0f);
    }

    // Run with a couple more settings.
    b.vvals(7.0f);
    l_y_val = 2;
    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {row.getName(), &row_val},
        {col.getName(), &col_val},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));
    result = b.get();
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 9.0f);
    }

    a.destroy();
    b.destroy();
    result.destroy();
}

TEST(LoweringGPU, MatrixGPUAddSingleBind) {
    auto inputDS = new const annot::MatrixGPU("input_con");
    auto outputDS = new const annot::MatrixGPU("output_con");

    annot::MatrixAddGPU add;
    Variable row("row");
    Variable col("col");
    Variable l_x("l_x");
    Variable l_y("l_y");

    Variable x("x");

    std::vector<Compose> c = {add[{
        {"x", x.bindToGrid(Grid::Property::BLOCK_ID_X)},
        {"row", row},
        {"col", col},
        {"l_x", l_x},
        {"l_y", l_y},
    }](inputDS, outputDS)};

    Pipeline p(c);
    p.at_device();
    Runner run(p);

    run.compile(Runner::Options{
        .filename = "test",
        .prefix = "/tmp",
        .include = " -I " + std::string(GERN_ROOT_DIR) + "/test/library/matrix/impl",
        .arch = std::string(GERN_CUDA_ARCH),
    });

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixGPU b(row_val, col_val, row_val);
    b.vvals(3.0f);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {row.getName(), &row_val},
        {col.getName(), &col_val},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    impl::MatrixCPU result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 5.0f);
    }

    a.destroy();
    b.destroy();
    result.destroy();
}

TEST(LoweringGPU, MatrixGPUAddDoubleBind) {
    auto inputDS = new const annot::MatrixGPU("input_con");
    auto outputDS = new const annot::MatrixGPU("output_con");

    annot::MatrixAddGPU add;
    Variable row("row");
    Variable col("col");
    Variable l_x("l_x");
    Variable l_y("l_y");

    Variable x("x");
    Variable y("y");

    std::vector<Compose> c = {
        add[{
            {"x", x.bindToGrid(Grid::Property::BLOCK_ID_X)},
            {"y", y.bindToGrid(Grid::Property::BLOCK_ID_Y)},
            {"row", row},
            {"col", col},
            {"l_x", l_x},
            {"l_y", l_y},
        }](inputDS, outputDS)};

    Pipeline p(c);
    p.at_device();
    Runner run(p);

    run.compile(Runner::Options{
        .filename = "test",
        .prefix = "/tmp",
        .include = " -I " + std::string(GERN_ROOT_DIR) + "/test/library/matrix/impl",
        .arch = std::string(GERN_CUDA_ARCH),
    });

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixGPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixGPU b(row_val, col_val, row_val);
    b.vvals(3.0f);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {row.getName(), &row_val},
        {col.getName(), &col_val},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    impl::MatrixCPU result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(result.data[i] == 5.0f);
    }

    a.destroy();
    b.destroy();
    result.destroy();
}