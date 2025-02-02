#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "gpu-matrix-const.h"

int main() {
    constexpr int64_t h = 32 * 4;
    constexpr int64_t w = 32 * 4;
    constexpr int64_t block_size = 32;
    constexpr int64_t num_cols_q = h / 32;

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
        {"col", col.bindToInt64(w)},
        {"stride", stride_val.bindToInt64(block_size)},
    }];
    auto sum_row_specialize = &sum_row[{
        {"col", col.bindToInt64(w)},
        {"stride", stride_val.bindToInt64(block_size)},
    }];

    Composable program = {
        Global((Tile(b["row"], row.bindToInt64(4)) || Grid::Unit::BLOCK_Y)(
            (Tile(b["col"], col_val.bindToInt64(w))(
                max_row_specialize->operator()(max_row_out, a),
                subtract_vec(max_row_out, a, sub_temp),
                exp_matrix(sub_temp, exp_temp),
                sum_row_specialize->operator()(sum_row_out, exp_temp),
                divide_vec(sum_row_out, exp_temp, b)

                    ))))};

    Runner run(program);
    Runner::Options options;

    options.include = "-I /home/manya/gern/apps/"
                      " -I /home/manya/gern/test/";
    options.arch = "89";

    run.compile(options);

    MatrixType in;
    in.ascending();
    MatrixType out;
    out.vvals(0.0f);

    run.evaluate({
        {a.getName(), &in},
        {b.getName(), &out},
    });

    auto cpu_result = out.get();
    auto a_result = in.get();
}