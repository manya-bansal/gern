
#include "compose/runner.h"
#include "gern_annot/functions.h"
#include "wrappers/adt.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {

    auto input = AbstractDataTypePtr(new const annot::ArrayGPU("input"));
    auto output = AbstractDataTypePtr(new const annot::FloatPtr("output"));
    auto temp = AbstractDataTypePtr(new const annot::ArrayGPU("temp", true));

    int thread_per_block = 2;

    Variable var_thrds_per_blk{"var_thrds_per_blk"};
    Variable t1{"t1"};

    annot::GlobalSum2 global_sum;
    annot::BlockReduceTake2 block_reduce_take2;

    Variable bound_k_dim = var_thrds_per_blk.bind(thread_per_block);
    Variable temp_size("temp_size");

    int temp_size_val = 4;
    temp_size = temp_size.bind(temp_size_val);

    auto block_reduce_take2_sp = &block_reduce_take2[{{"block_size",
                                                       bound_k_dim}}];
    auto global_sum_sp = &global_sum[{{"k", temp_size}}];

    Composable program = {
        Global(
            (Reduce(temp_size, t1.bind(1)) || Grid::Unit::BLOCK_X)(
                (*block_reduce_take2_sp)(temp, input),
                global_sum(output, temp)),
            {{Grid::BLOCK_DIM_X, var_thrds_per_blk.bind(thread_per_block)}}),
    };

    Runner::Options options;
    options.filename = "gern_hello_cccl.cu";
    options.cpp_std = "c++14";  // cccl requires c++14
    options.arch = "89";
    options.include = " -I/home/manya/gern/gernels/cccl";

    Runner runner(program);
    runner.compile(options);

    // Let's try running now...
    int size_of_output = 1;
    int size_of_input = size_of_output * temp_size_val * thread_per_block;

    ArrayGPU output_real(size_of_output);
    output_real.vvals(0.0f);
    ArrayGPU input_real(size_of_input);
    input_real.ascending();

    runner.evaluate({{input.getName(), &input_real},
                     {output.getName(), &output_real.data}});

    auto output_cpu = output_real.get();
    assert(output_cpu.data[0] == size_of_input * (size_of_input - 1) / 2);
}
