#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "impl/gpu-matrix-const.h"

#include <iostream>

using namespace gern;

int main() {
    constexpr int64_t row_val = 8;
    constexpr int64_t col_val = 8;

    constexpr int64_t size_y_chain = 1;
    constexpr int64_t size_x_chain = 1;
    constexpr int64_t stride_val = 3;

    // This needs to be a multiple of 4 to work with vectorization.
    constexpr int64_t row_val_in = row_val + stride_val + (size_y_chain - 2) + 2;  // Want to pad by a multiple of four.
    constexpr int64_t col_val_in = col_val + stride_val + (size_x_chain - 2) + 2;  // Want to pad by a multiple of four.;

    constexpr int64_t block_size = 1;

    using OutputType = impl::MatrixGPU<row_val, col_val, col_val, block_size>;
    using OutputTypeAnnot = annot::MatrixGPUSequential<row_val, col_val, block_size>;
    using InputType = impl::MatrixGPU<row_val_in, col_val_in, col_val_in, block_size>;
    using InputTypeAnnot = annot::MatrixGPUSequential<row_val_in, col_val_in, block_size>;

    auto input = AbstractDataTypePtr(new const InputTypeAnnot("input", false));
    auto output = AbstractDataTypePtr(new const OutputTypeAnnot("output", false));
    auto temp = AbstractDataTypePtr(new const OutputTypeAnnot("temp", true));  // The type of this guy will not matter.

    Variable col("col");
    Variable row("row");
    Variable stride{"stride"};

    annot::BlurX blur_x_no_template{input, temp};
    auto blur_x_1 = &blur_x_no_template[{
        {"stride", stride.bind(stride_val)},
    }];
    annot::BlurY blur_x_no_template_2{input, temp};
    auto blur_x_2 = &blur_x_no_template_2[{
        {"stride", stride.bind(stride_val)},
    }];

    Composable program = {
        Global(
            Tile(output["col"], col.bind(4))(
                Tile(output["row"], row.bind(4))(
                    blur_x_1->operator()(input, temp),
                    blur_x_2->operator()(temp, output))))};

    Runner run(program);
    Runner::Options options;
    options.filename = "blur.cu";
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

    std::cout << "CPU result: " << cpu_result << std::endl;

    return 0;
}