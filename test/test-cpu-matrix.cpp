#include "compose/composable.h"
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
    Variable l_x("l_x");
    Variable l_y("l_y");

    Composable program = {
        (Tile(outputDS["row"], l_x))(
            Tile(outputDS["col"], l_y)(
                add(inputDS, outputDS)))};

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t row_val = 10;
    int64_t col_val = 20;

    impl::MatrixCPU a(row_val, col_val, col_val);
    a.vvals(2.0f);
    impl::MatrixCPU b(row_val, col_val, col_val);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(b.data[i] == 3.0f);
    }

    // Run with a couple more settings.
    a.vvals(7.0f);
    l_y_val = 2;
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(b.data[i] == 8.0f);
    }

    a.destroy();
    b.destroy();
}

// TEST(LoweringCPU, Softmax) {

//     auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU("input"));
//     auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU("output_final"));
//     auto ExpDS = AbstractDataTypePtr(new const annot::MatrixCPU("expDS"));
//     auto SubDS = AbstractDataTypePtr(new const annot::MatrixCPU("subDS"));
//     auto SumRowDS = AbstractDataTypePtr(new const annot::ArrayCPU("rowSum"));
//     auto MaxRowDS = AbstractDataTypePtr(new const annot::ArrayCPU("rowMax"));

//     annot::SumRow sum_row;
//     annot::MaxRow max_row;
//     annot::SubtractVec subtract_vec;
//     annot::ExpMatrix exp_matrix;
//     annot::DivideVec divide_vec;

//     Variable row("row");
//     Variable col("col");
//     Variable col_1("col_1");
//     Variable l_x("l_x");
//     Variable l_y("l_y");

//     std::vector<Compose> c = {
//         max_row[{
//             {"col", col},  // This should go away, once I allow member variables as args.
//         }](inputDS, MaxRowDS),
//         subtract_vec(MaxRowDS, inputDS, SubDS),
//         exp_matrix(SubDS, ExpDS),
//         sum_row[{
//             {"col", col},  // This should go away, once I allow member variables as args.
//         }](ExpDS, SumRowDS),
//         divide_vec[{
//             {"l_x", l_x},
//             {"l_y", l_y},
//         }](SumRowDS, ExpDS, outputDS),
//     };

//     Pipeline p(c);
//     Runner run(p);

//     run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

//     int64_t row_val = 2;
//     int64_t col_val = 2;
//     int64_t l_x_val = 2;
//     int64_t l_y_val = col_val;

//     impl::MatrixCPU a(row_val, col_val, row_val);
//     // fill with random values.
//     a.random_fill();
//     impl::MatrixCPU b(row_val, col_val, row_val);
//     b.vvals(0.0f);

//     ASSERT_NO_THROW(run.evaluate({
//         {inputDS.getName(), &a},
//         {outputDS.getName(), &b},
//         {col.getName(), &col_val},
//         {l_x.getName(), &l_x_val},
//         {l_y.getName(), &l_y_val},
//     }));

//     // Compute unfused for reference.
//     impl::MatrixCPU reference(row_val, col_val, row_val);
//     reference.vvals(0.0f);
//     impl::ArrayCPU max_row_ref(row_val);

//     gern::impl::max_row(a, max_row_ref);
//     gern::impl::subtract_vec(max_row_ref, a, reference);
//     gern::impl::exp_matrix(reference, reference);
//     gern::impl::sum_row(reference, max_row_ref);
//     gern::impl::divide_vec(max_row_ref, reference, reference);

//     for (int i = 0; i < row_val * col_val; i++) {
//         ASSERT_EQ(b.data[i], reference.data[i]);
//     }

//     a.destroy();
//     b.destroy();
//     reference.destroy();
//     max_row_ref.destroy();
// }

TEST(LoweringCPU, MatrixCPUTranspose) {
    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU("input"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU("output"));

    annot::MatrixTransposeCPU transpose;
    Variable l_x("l_x");
    Variable l_y("l_y");

    Composable program = {
        (Tile(outputDS["row"], l_x))(
            Tile(outputDS["col"], l_y)(
                transpose(inputDS, outputDS)))};

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t row_val_input = 21;
    int64_t col_val_input = 15;
    impl::MatrixCPU a(row_val_input, col_val_input, col_val_input);
    a.ascending();
    int64_t row_val_output = 15;
    int64_t col_val_output = 21;
    impl::MatrixCPU b(row_val_output, col_val_output, col_val_output);

    int64_t l_x_val = 5;
    int64_t l_y_val = 7;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
    }));

    for (int i = 0 ; i < row_val_input ; i ++ ) {
        for (int j = 0 ; j < col_val_input ; j ++ ) {
            ASSERT_EQ(a(i,j), b(j,i)); // note: a(i,j) = a.data[i * a.lda + j]
        }
    }

    a.destroy();
    b.destroy();
}