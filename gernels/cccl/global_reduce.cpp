
#include "../current_path.h"
#include "compose/runner.h"
#include "gern_annot/functions.h"
#include "wrappers/adt.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {

    annot::GridReduce grid_reduce;
    auto input = AbstractDataTypePtr(new const annot::ArrayGPU("input"));
    auto output = AbstractDataTypePtr(new const annot::FloatPtr("output"));
    // auto output = AbstractDataTypePtr(new const annot::ArrayGPU("output"));

    Variable k{"k"};
    Variable block_size{"block_size"};
    Variable input_size{"input_size"};

    int k_val = 10;
    int num_threads_p_blk = 2;

    // Specialize the function to the input size.
    input_size = input_size.bind(k_val);
    auto grid_reduce_sp = &grid_reduce[{
        {"k", input_size},
    }];

    // Variable for tile size of the reduction.
    Variable elem_per_thread = block_size.bind(num_threads_p_blk);
    Composable program = {
        Global(
            (Reduce(input_size, elem_per_thread) || Grid::Unit::BLOCK_X)(
                (*grid_reduce_sp)(output, input)),
            {{Grid::BLOCK_DIM_X, elem_per_thread}}),
    };

    Runner::Options options;
    options.filename = "hello_cccl.cu";
    options.cpp_std = "c++14";  // cccl requires c++14 or higher.
    options.arch = GERNELS_ARCH;
    options.include = " -I" + std::string(GERNELS_PATH) + "/cccl";

    Runner runner(program);
    runner.compile(options);

    // Let's try running now...
    ArrayGPU input_real(k_val);
    input_real.ascending();
    ArrayGPU output_real(1);
    output_real.vvals(0.0f);

    runner.evaluate({{input.getName(), &input_real},
                     {output.getName(), &output_real.data}});

    auto output_cpu = output_real.get();
    assert(output_cpu.data[0] == k_val * (k_val - 1) / 2);

    input_real.destroy();
    output_real.destroy();
    output_cpu.destroy();
}
