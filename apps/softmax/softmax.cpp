#include "annot/adt.h"
#include "annot/functions.h"
#include "benchmark.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "impl/gpu-matrix-const.h"

int main() {
    constexpr int64_t h = 16384;
    constexpr int64_t w = 16384;
    constexpr int64_t block_size = 1024;
    constexpr int64_t num_cols_q = w / block_size;

    using MatrixType = impl::MatrixGPU<h, w, h, block_size>;

    auto a = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("input", false));
    auto max_row_out = AbstractDataTypePtr(new const annot::StaticArray("max_row_out"));
    auto sum_row_out = AbstractDataTypePtr(new const annot::StaticArray("sum_row_out"));
    auto b = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("output", false));
    auto sub_temp = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("sub_temp", true));
    auto div_temp = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("div_temp", true));
    auto exp_temp = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("exp_temp", true));

    annot::ExpMatrix exp_matrix{a, b};
    annot::MaxRow max_row{max_row_out, a};
    annot::SumRow sum_row{sum_row_out, exp_temp};
    annot::SubstractVec subtract_vec{max_row_out, a, sub_temp};
    annot::DivideVec divide_vec{sum_row_out, exp_temp, b};

    Variable row("row");
    Variable col("col");
    Variable col_val("col_val");
    Variable stride_val("stride_val");

    Variable col_bound = col.bindToInt64(num_cols_q);
    auto max_row_specialize = &max_row[{
        {"col", col.bindToInt64(num_cols_q)},
        {"stride", stride_val.bindToInt64(num_cols_q)},
    }];
    auto sum_row_specialize = &sum_row[{
        {"col", col.bindToInt64(num_cols_q)},
        {"stride", stride_val.bindToInt64(num_cols_q)},
    }];

    Composable program = {
        Global((Tile(b["row"], row.bindToInt64(1)) || Grid::Unit::BLOCK_X)(
                   ((Tile(b["col"], col_val.bindToInt64(num_cols_q)) || Grid::Unit::THREAD_X)(
                       ((Reduce(a["col"], col_val.bindToInt64(num_cols_q))) || Grid::Unit::THREAD_Y)(
                           max_row_specialize->operator()(max_row_out, a)),
                       subtract_vec(max_row_out, a, sub_temp),
                       exp_matrix(sub_temp, exp_temp),
                       ((Reduce(exp_temp["col"], col_val.bindToInt64(num_cols_q))) || Grid::Unit::THREAD_Y)(
                           sum_row_specialize->operator()(sum_row_out, exp_temp)),
                       divide_vec(sum_row_out, exp_temp, b)))),
               {
                   {Grid::Dim::BLOCK_DIM_Y, stride_val.bindToInt64(block_size)},
               })};

    Runner run(program);
    Runner::Options options;

    options.include = "-I /home/manya/gern/apps/common"
                      " -I /home/manya/gern/test/";
    options.arch = "89";

    run.compile(options);

    MatrixType in;
    in.ascending();
    MatrixType out;
    out.vvals(0.0f);

    auto func = [&]() {
        run.evaluate({
            {a.getName(), &in},
            {b.getName(), &out},
        });
    };

    auto time = my_benchmark::benchmark(5, 5, func);

    double gflops = sizeof(float) * h * w * 2 * 1e-9;

    std::cout << (gflops / (time / 5)) << std::endl;

    auto cpu_result = out.get();
    auto a_result = in.get();
}