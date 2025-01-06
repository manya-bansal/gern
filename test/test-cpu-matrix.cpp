#include "annotations/visitor.h"
#include "codegen/runner.h"
#include "compose/compose.h"
#include "compose/pipeline.h"

#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"

#include "config.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringCPU, MatrixCPUAdd) {
    auto inputDS = std::make_shared<const annot::MatrixCPU>("input_con");
    auto outputDS = std::make_shared<const annot::MatrixCPU>("output_con");

    annot::MatrixAddCPU add;
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
    Runner run(p);

    run.compile(Runner::Options{
        "nvcc",
        "test_matrix.cpp",
        "/tmp",
        " -I " + std::string(GERN_ROOT_DIR) + "/test/library/matrix/impl",
        "",
    });

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixCPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixCPU b(row_val, col_val, row_val);
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

    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(b.data[i] == 5.0f);
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
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(b.data[i] == 9.0f);
    }

    a.destroy();
    b.destroy();
}
