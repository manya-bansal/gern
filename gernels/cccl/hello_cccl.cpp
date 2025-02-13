
#include "compose/runner.h"
#include "gern_annot/functions.h"
#include "wrappers/adt.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {

    annot::GlobalSum global_sum;
    annot::BlockReduce block_reduce;
    auto input = AbstractDataTypePtr(new const annot::ArrayGPU("input"));
    auto output = AbstractDataTypePtr(new const annot::FloatPtr("output"));
    // auto output = AbstractDataTypePtr(new const annot::ArrayGPU("output"));

    Variable k{"k"};
    Variable block_size{"block_size"};

    int k_val = 10;

    int num_threads_p_blk = 2;
    auto k_const = k.bind(10);

    auto k_const_2 = k.bind(2);

    auto elem_per_thread = block_size.bind(num_threads_p_blk);

    auto global_sum_sp = &global_sum[{{"k", k_const}}];

    auto block_reduce_sp = &block_reduce[{
        {"k", k_const_2},
    }];

    Composable program = {
        Global(
            (Reduce(input["size"], elem_per_thread) || Grid::Unit::BLOCK_X)(
                (*block_reduce_sp)(output, input)),
            {{Grid::BLOCK_DIM_X, elem_per_thread}}),
    };

    Runner::Options options;
    options.filename = "gern_hello_cccl.cu";
    options.cpp_std = "c++14";  // cccl requires c++14
    options.arch = "89";
    options.include = " -I/home/manya/gern/gernels/cccl";

    Runner runner(program);
    runner.compile(options);

    // Let's try running now...
    ArrayGPU input_real(k_val);
    input_real.ascending();
    ArrayGPU output_real(2);
    output_real.vvals(0.0f);

    runner.evaluate({{input.getName(), &input_real},
                     {output.getName(), &output_real.data}});

    auto output_cpu = output_real.get();
    assert(output_cpu.data[0] == k_val * (k_val - 1) / 2);

    input_real.destroy();
    output_real.destroy();
    output_cpu.destroy();
}
