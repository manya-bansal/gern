#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "impl/gpu-matrix-const.h"

#include <iostream>

using namespace gern;

int main() {
    constexpr int64_t row_val = 4;
    constexpr int64_t col_val = 4;

    constexpr int64_t size_x_chain = 2;
    constexpr int64_t stride_val = 3;

    constexpr int64_t row_val_in = 4;  // Want to pad by a multiple of four.
    constexpr int64_t col_val_in = 4 + stride_val + (size_x_chain - 2) + 4;

    constexpr int64_t block_size = 1;

    using OutputType = impl::MatrixGPU<row_val, col_val, row_val, block_size>;
    using OutputTypeAnnot = annot::MatrixGPUSequential<row_val, col_val, block_size>;
    using InputType = impl::MatrixGPU<row_val_in, col_val_in, row_val_in, block_size>;
    using InputTypeAnnot = annot::MatrixGPUSequential<row_val_in, col_val_in, block_size>;

    auto input = AbstractDataTypePtr(new const InputTypeAnnot("input", false));
    auto output = AbstractDataTypePtr(new const OutputTypeAnnot("output", false));
    auto temp = AbstractDataTypePtr(new const OutputTypeAnnot("temp", true));  // The type of this guy will not matter.
    Variable col("col");
    Variable row("row");
    Variable stride{"stride"};

    annot::BlurX blur_x_no_template{input, temp};
    auto blur_x_1 = &blur_x_no_template[{
        {"stride", stride.bindToInt64(stride_val)},
    }];
    annot::BlurX blur_x_no_template_2{input, temp};
    auto blur_x_2 = &blur_x_no_template_2[{
        {"stride", stride.bindToInt64(stride_val)},
    }];

    Composable program = {
        Global(
            Tile(output["col"], col.bindToInt64(col_val))(
                Tile(output["row"], row.bindToInt64(row_val))(
                    blur_x_1->operator()(input, temp),
                    blur_x_2->operator()(temp, output)

                        )))};

    Runner run(program);
    Runner::Options options;
    options.include = "-I /home/manya/gern/apps/common"
                      " -I /home/manya/gern/test/";
    options.arch = "89";
    run.compile(options);

    InputType in;
    in.ascending();
    OutputType out;
    out.vvals(0.0f);

    run.evaluate({
        {input.getName(), &in},
        {output.getName(), &out},
    });

    auto cpu_result = out.get();
    auto a_result = in.get();
    std::cout << cpu_result << std::endl;
    std::cout << a_result << std::endl;

    return 0;
}