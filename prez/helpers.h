#pragma once

#include "compose/composable.h"
// #include "compose/composable.h"
// #include "compose/runner.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/annot/gpu-array.h"
#include "library/matrix/annot/cpu-matrix.h"

using namespace gern;

inline AbstractDataTypePtr mk_array(const std::string &name) {
    return AbstractDataTypePtr(new annot::ArrayCPU(name));
}

inline AbstractDataTypePtr mk_matrix(const std::string &name) {
    return AbstractDataTypePtr(new annot::MatrixCPU(name));
}

inline AbstractDataTypePtr mk_array_gpu(const std::string &name) {
    return AbstractDataTypePtr(new annot::ArrayGPU(name));
}

inline Runner compile_program(Composable program,
                              std::string name = "program.cpp") {
    Runner runner(program);
    Runner::Options options{
        .filename = name,
        .prefix = "/home/manya/gern/prez/gern_gen",
        .include = " -I /home/manya/gern/prez/library/array/impl"
                   " -I /home/manya/gern/prez/library/matrix/impl",
    };
    runner.compile(options);
    return runner;
}

inline Runner compile_program_gpu(Composable program,
                                  std::string name = "program.cu") {
    Runner runner(program);
    Runner::Options options{
        .filename = name,
        .prefix = "/home/manya/gern/prez/gern_gen",
        .include = " -I /home/manya/gern/prez/library/array/impl"
                   " -I /home/manya/gern/prez/library/matrix/impl",
        .arch = "89",
    };
    runner.compile(options);
    return runner;
}

class TrivialManager : public grid::SharedMemoryManager {
public:
    TrivialManager(Variable smem_size)
        : grid::SharedMemoryManager(
              FunctionCall{"init_shmem",
                           {smem_size},
                           {},
                           Parameter(),
                           LaunchArguments(),
                           LaunchArguments(),
                           DEVICE},
              {
                  "/home/manya/gern/test/library/smem_allocator/sh_malloc.h",
              }) {
    }
};