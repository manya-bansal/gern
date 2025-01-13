#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringCPU, MatrixCPUAdd) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU("output_con"));

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

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t row_val = 10;
    int64_t col_val = 10;

    impl::MatrixCPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixCPU b(row_val, col_val, row_val);
    b.vvals(3.0f);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
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
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
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

TEST(LoweringCPU, Softmax) {

    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU("input"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU("output_final"));
    auto SumRowDS = AbstractDataTypePtr(new const annot::ArrayCPU("rowSum"));
    auto MaxRowDS = AbstractDataTypePtr(new const annot::ArrayCPU("rowMax"));

    annot::SumRow sum_row;
    annot::MaxRow max_row;
    annot::SubtractVec subtract_vec;

    Variable row("row");
    Variable col("col");
    Variable col_1("col_1");
    Variable l_x("l_x");
    Variable l_y("l_y");

    std::vector<Compose> c = {
        max_row[{
            {"col", col_1},  // This should go away, once I allow member variables as args.
        }](inputDS, MaxRowDS),
        subtract_vec[{
            {"row", row},
            {"col", col},
            {"l_x", l_x},
            {"l_y", l_y},
        }](MaxRowDS, inputDS, outputDS),
    };

    Pipeline p(c);
    Runner run(p);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t row_val = 2;
    int64_t col_val = 2;
    int64_t l_x_val = 2;
    int64_t l_y_val = col_val;

    impl::MatrixCPU a(row_val, col_val, row_val);
    // fill with random values.
    a.random_fill();
    impl::MatrixCPU b(row_val, col_val, row_val);
    b.vvals(0.0f);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {row.getName(), &row_val},
        {col.getName(), &col_val},
        {col_1.getName(), &col_val},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    // Compute unfused for reference.
    impl::MatrixCPU reference(row_val, col_val, row_val);
    reference.vvals(0.0f);
    impl::ArrayCPU max_row_ref(row_val);

    gern::impl::max_row(a, max_row_ref);
    gern::impl::subtract_vec(max_row_ref, a, reference);

    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_EQ(b.data[i], reference.data[i]);
    }

    a.destroy();
    b.destroy();
    reference.destroy();
    max_row_ref.destroy();
}