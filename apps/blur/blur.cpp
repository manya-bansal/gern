#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "impl/gpu-matrix-const.h"

#include <iostream>

using namespace gern;

void cpu_blur_x(impl::MatrixCPU &input, impl::MatrixCPU &output) {
    for (int64_t i = 0; i < output.row; i++) {
        for (int64_t j = 0; j < output.col; j++) {
            output(i, j) = (input(i, j) + input(i, j + 1) + input(i, j + 2)) / 3;
        }
    }
}

void cpu_blur_y(impl::MatrixCPU &input, impl::MatrixCPU &output) {
    for (int64_t i = 0; i < output.row; i++) {
        for (int64_t j = 0; j < output.col; j++) {
            output(i, j) = (input(i, j) + input(i + 1, j) + input(i + 2, j)) / 3;
        }
    }
}

int main() {
    constexpr int64_t row_val = 128 * 90;
    constexpr int64_t col_val = 128 * 90;

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
    Variable col_inner("col_inner");
    Variable row("row");
    Variable row_inner("row_inner");
    Variable stride{"stride"};

    annot::BlurX blur_x_no_template{input, temp};
    auto blur_x_t = &blur_x_no_template[{
        {"stride", stride.bind(stride_val)},
    }];

    annot::BlurY blur_y{input, temp};
    auto blur_y_t = &blur_y[{
        {"stride", stride.bind(stride_val)},
    }];

    Composable program = {
        Global(
            (Tile(output["col"], col.bind(128)) || Grid::BLOCK_X)(
                (Tile(output["row"], row.bind(128)) || Grid::BLOCK_Y)(
                    (Tile(output["col"], col_inner.bind(8)) || Grid::THREAD_X)(
                        (Tile(output["row"], row_inner.bind(8)) || Grid::THREAD_Y)(
                            (*blur_x_t)(input, temp),
                            (*blur_y_t)(temp, output))))))};

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

    auto gern_result = out.get();
    auto in_result = in.get();

    // Compute a reference result on the CPU.
    auto blur_result_x = impl::MatrixCPU(row_val + 2, col_val, col_val);
    auto blur_result_y = impl::MatrixCPU(row_val, col_val, col_val);

    cpu_blur_x(in_result, blur_result_x);
    cpu_blur_y(blur_result_x, blur_result_y);

    // Make sure the result is correct.
    for (int64_t i = 0; i < row_val; i++) {
        for (int64_t j = 0; j < col_val; j++) {
            assert(gern_result(i, j) == blur_result_y(i, j));
        }
    }

    std::cout << "Values are equal" << std::endl;

    blur_result_y.destroy();
    blur_result_x.destroy();
    in_result.destroy();
    out.destroy();
    in.destroy();
    gern_result.destroy();

    return 0;
}