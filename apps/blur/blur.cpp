#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "impl/gpu-matrix-const.h"

#include <iostream>

using namespace gern;

int main() {
    constexpr int64_t h = 32 * 4;
    constexpr int64_t w = 32 * 4;
    constexpr int64_t block_size = 32;
    constexpr int64_t num_cols_q = h / 32;

    using MatrixType = impl::MatrixGPU<h, w, h, block_size>;

    auto input = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("input", false));
    auto output = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("output", false));
    auto temp = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, block_size>("temp", true));
    Variable col("col");
    Variable row("row");
    Variable stride{"stride"};

    annot::BlurX blur_x_no_template{input, temp};
    auto blur_x_1 = &blur_x_no_template[{
        {"stride", stride.bindToInt64(3)},
    }];
    annot::BlurX blur_x_no_template_2{input, temp};
    auto blur_x_2 = &blur_x_no_template_2[{
        {"stride", stride.bindToInt64(3)},
    }];

    Composable program = {
        Global(
            Tile(output["col"], col.bindToInt64(w))(
                Tile(output["row"], row.bindToInt64(h))(
                    blur_x_1->operator()(input, temp),
                    blur_x_2->operator()(temp, output))))};

    Runner run(program);
    Runner::Options options;
    options.include = "-I /home/manya/gern/apps/common"
                      " -I /home/manya/gern/test/";
    options.arch = "89";
    run.compile(options);

    return 0;
}