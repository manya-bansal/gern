#include "compose/composable.h"
#include "compose/runner.h"
#include "config.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>

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
    int64_t col_val = 10;

    impl::MatrixCPU a(row_val, col_val, row_val);
    a.vvals(2.0f);
    impl::MatrixCPU b(row_val, col_val, row_val);

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

TEST(LoweringCPU, Attention) {
    auto q = AbstractDataTypePtr(new const annot::MatrixCPU("q"));
    auto k = AbstractDataTypePtr(new const annot::MatrixCPU("k"));
    auto v = AbstractDataTypePtr(new const annot::MatrixCPU("v"));
    
    auto kt = AbstractDataTypePtr(new const annot::MatrixCPU("kt"));
    auto q_kt = AbstractDataTypePtr(new const annot::MatrixCPU("q_kt"));
    auto sm_in = AbstractDataTypePtr(new const annot::MatrixCPU("sm_in"));
    auto sm_out = AbstractDataTypePtr(new const annot::MatrixCPU("sm_out"));
    
    auto output = AbstractDataTypePtr(new const annot::MatrixCPU("output"));

    Variable l_x("l_x");
    Variable l_y("l_y");

    Variable q_width("q_width");
    Variable q_height("q_height");
    Variable sqrt_dk("sqrt_dk", Datatype::Float32);

    annot::MatrixTranspose transpose;    
    annot::MatrixMultiply matmul1;
    annot::MatrixMultiply matmul2;
    annot::MatrixDivn divn;
    annot::MatrixSoftmax softmax;

    auto matmul1_specialize = &matmul1[{
        {"shared_len", q_width}
    }];

    auto matmul2_specialize = &matmul2[{
        {"shared_len", q_height}
    }];

    Composable program = {
        Tile(output["row"], l_x)(
            Tile(output["col"], l_y)(
                transpose(k, kt),
                matmul1(q, kt, q_kt),
                divn(q_kt, sqrt_dk, sm_in),
                softmax(sm_in, sm_out),
                matmul2(sm_out, v, output)
            )
        )
    };

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));
    
    int64_t nk = 1024;
    int64_t dk = 64;
    int64_t l_x_val = 256;
    int64_t l_y_val = dk;

    float sqrt_dk_val = sqrt(dk);

    impl::MatrixCPU in_q(nk, dk, dk);
    impl::MatrixCPU in_k(nk, dk, dk);
    impl::MatrixCPU in_v(nk, dk, dk);
    impl::MatrixCPU out(nk, dk, dk);

    in_q.random_fill();
    in_k.random_fill();
    in_v.random_fill();

    impl::MatrixCPU ref_kt(dk, nk, nk);
    impl::MatrixCPU ref_q_kt(nk, nk, nk);
    impl::MatrixCPU ref_sm_in(nk, nk, nk);
    impl::MatrixCPU ref_sm_out(nk, nk, nk);
    impl::MatrixCPU ref_out(nk, dk, dk);

    gern::impl::transpose(in_k, ref_kt);
    gern::impl::mmul(in_q, ref_kt, ref_q_kt);
    gern::impl::divn(ref_q_kt, sqrt_dk_val, ref_sm_in);
    gern::impl::softmax(ref_sm_in, ref_sm_out);
    gern::impl::mmul(ref_sm_out, in_v, ref_out);

    ASSERT_NO_THROW(run.evaluate({
        {q.getName(), &in_q},
        {k.getName(), &in_k},
        {v.getName(), &in_v},
        {sqrt_dk.getName(), &sqrt_dk_val},
        {output.getName(), &out},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
        {q_width.getName(), &dk},
        {q_height.getName(), &nk},
    }));

    for (int i = 0; i < nk * dk; i++) {
        ASSERT_EQ(out.data[i], ref_out.data[i]);
    }
}

TEST(LoweringCPU, SoftmaxEnd) {
    auto q = AbstractDataTypePtr(new const annot::MatrixCPU("q"));
    auto k = AbstractDataTypePtr(new const annot::MatrixCPU("k"));
    auto v = AbstractDataTypePtr(new const annot::MatrixCPU("v"));
    
    auto kt = AbstractDataTypePtr(new const annot::MatrixCPU("kt"));
    auto q_kt = AbstractDataTypePtr(new const annot::MatrixCPU("q_kt"));
    auto sm_in = AbstractDataTypePtr(new const annot::MatrixCPU("sm_in"));
    
    auto output = AbstractDataTypePtr(new const annot::MatrixCPU("output"));

    Variable l_x("l_x");
    Variable l_y("l_y");

    Variable q_width("q_width");
    Variable sqrt_dk("sqrt_dk", Datatype::Float32);

    annot::MatrixTranspose transpose;    
    annot::MatrixMultiply matmul1;
    annot::MatrixDivn divn;
    annot::MatrixSoftmax softmax;

    auto matmul1_specialize = &matmul1[{
        {"shared_len", q_width}
    }];
    Composable program = {
        Tile(output["row"], l_x)(
			Tile(output["col"], l_y)(
				transpose(k, kt),
				matmul1(q, kt, q_kt),
				divn(q_kt, sqrt_dk, sm_in),
				softmax(sm_in, output)
			)
        )
    };

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));
    
    int64_t nk = 1024;
    int64_t dk = 64;
    int64_t l_x_val = 256;
    int64_t l_y_val = nk;
	int64_t zero_val = 0;

    float sqrt_dk_val = sqrt(dk);

    impl::MatrixCPU in_q(nk, dk, dk);
    impl::MatrixCPU in_k(nk, dk, dk);
    impl::MatrixCPU in_v(nk, dk, dk);
    impl::MatrixCPU out(nk, nk, nk);

    in_q.random_fill();
    in_k.random_fill();
    in_v.random_fill();

    impl::MatrixCPU ref_kt(dk, nk, nk);
    impl::MatrixCPU ref_q_kt(nk, nk, nk);
    impl::MatrixCPU ref_sm_in(nk, nk, nk);
    impl::MatrixCPU ref_out(nk, nk, nk);

    gern::impl::transpose(in_k, ref_kt);
    gern::impl::mmul(in_q, ref_kt, ref_q_kt);
    gern::impl::divn(ref_q_kt, sqrt_dk_val, ref_sm_in);
    gern::impl::softmax(ref_sm_in, ref_out);

    ASSERT_NO_THROW(run.evaluate({
        {q.getName(), &in_q},
        {k.getName(), &in_k},
        {sqrt_dk.getName(), &sqrt_dk_val},
        {output.getName(), &out},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
        {q_width.getName(), &dk},
    }));

    for (int i = 0; i < nk * nk; i++) {
        ASSERT_EQ(out.data[i], ref_out.data[i]);
    }
}


TEST(LoweringCPU, Matmul) {
    auto inA = AbstractDataTypePtr(new const annot::MatrixCPU("a"));
    auto inB = AbstractDataTypePtr(new const annot::MatrixCPU("b"));	
    auto output = AbstractDataTypePtr(new const annot::MatrixCPU("output"));	

    annot::MatrixMultiply matmul;

    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable shared_len("shared_len");

    auto matmul_specialize = &matmul[{
        {"shared_len", shared_len}
    }];

    Composable program = {
        Tile(output["row"], l_x)(
            Tile(output["col"], l_y)(
                matmul_specialize->operator()(inA, inB, output)
            )
        )
    };

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    int64_t m = 10;
    int64_t n = 20;
    int64_t k = 15;

    impl::MatrixCPU a(m, k, k);
    a.random_fill();
    impl::MatrixCPU b(k, n, n);
    b.random_fill();
    impl::MatrixCPU out(m, n, n);
    impl::MatrixCPU reference(m, n, n);

    int64_t l_x_val = 5;
    int64_t l_y_val = 5;

    gern::impl::mmul(a, b, reference);

    ASSERT_NO_THROW(run.evaluate({
        {inA.getName(), &a},
        {inB.getName(), &b},
        {output.getName(), &out},
        {l_x.getName(), &l_x_val},
        {l_y.getName(), &l_y_val},
        {shared_len.getName(), &k}
    }));
    
    for (int i = 0; i < m * n; i++) {
        ASSERT_EQ(out.data[i], reference.data[i]);
    }
}


TEST(LoweringCPU, Softmax) {
    auto sm_in = AbstractDataTypePtr(new const annot::MatrixCPU("sm_in"));
    auto sm_out = AbstractDataTypePtr(new const annot::MatrixCPU("sm_out"));

    int64_t row_val = 10;
    int64_t col_val = 20;

    Variable l_x("l_x");
    Variable l_y("l_y");
    Variable y("y");

    annot::MatrixSoftmax softmax;
    auto softmax_specialize = &softmax[{
        {"y", y.bind(0)},
        {"l_y", l_y.bind(col_val)}
    }];

    Composable program = {
        Tile(sm_out["row"], l_x)(
            softmax(sm_in, sm_out)
        )
    };

    Runner run(program);

    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));
    

    impl::MatrixCPU a(row_val, col_val, col_val);
    a.random_fill();
    impl::MatrixCPU b(row_val, col_val, col_val);

    impl::MatrixCPU reference(row_val, col_val, col_val);
    gern::impl::softmax(a, reference);

    int64_t l_x_val = 5;

    ASSERT_NO_THROW(run.evaluate({
        {sm_in.getName(), &a},
        {sm_out.getName(), &b},
        {l_x.getName(), &l_x_val},
    }));

    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_EQ(b.data[i], reference.data[i]);
    }
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