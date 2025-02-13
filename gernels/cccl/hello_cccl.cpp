
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
    auto output = AbstractDataTypePtr(new const annot::ArrayGPU("output"));

    int thread_per_block = 2;

    Variable var_thrds_per_blk{"var_thrds_per_blk"};
    Variable t1{"t1"};

    annot::BlockReduceTake2 block_reduce_take2;
    auto block_reduce_take2_sp = &block_reduce_take2[{{"block_size",
                                                       var_thrds_per_blk.bind(thread_per_block)}}];

    Composable program = {
        Global(
            (Tile(output["size"], t1.bind(1)) || Grid::Unit::BLOCK_X)(
                (*block_reduce_take2_sp)(output, input)),
            {{Grid::BLOCK_DIM_X, var_thrds_per_blk.bind(thread_per_block)}}),
    };

    Runner::Options options;
    options.filename = "gern_hello_cccl.cu";
    options.cpp_std = "c++14";  // cccl requires c++14
    options.arch = "89";
    options.include = " -I/home/manya/gern/gernels/cccl";

    Runner runner(program);
    runner.compile(options);

    // // Let's try running now...
    int size_of_output = 4;
    ArrayGPU output_real(size_of_output);
    output_real.vvals(0.0f);
    ArrayGPU input_real(size_of_output * thread_per_block);
    input_real.ascending();

    runner.evaluate({{input.getName(), &input_real},
                     {output.getName(), &output_real}});

    auto output_cpu = output_real.get();
    std::cout << "output_cpu: " << output_cpu.data[0] << std::endl;
    std::cout << "output_cpu: " << output_cpu.data[1] << std::endl;
    std::cout << "output_cpu: " << output_cpu.data[2] << std::endl;
    std::cout << "output_cpu: " << output_cpu.data[3] << std::endl;

    // assert(output_cpu.data[0] == k_val * (k_val - 1) / 2);

    // input_real.destroy();
    // output_real.destroy();
    // output_cpu.destroy();
}
