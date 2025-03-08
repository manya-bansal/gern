#pragma once

#include "compose/composable.h"
// #include "compose/composable.h"
// #include "compose/runner.h"
#include "library/array/annot/cpu-array.h"

using namespace gern;

inline AbstractDataTypePtr mk_array(const std::string &name) {
    return AbstractDataTypePtr(new annot::ArrayCPU(name));
}

inline Runner compile_program(Composable program,
                              std::string name = "program.cpp") {
    Runner runner(program);
    Runner::Options options{
        .filename = name,
        .prefix = "/home/manya/gern/prez/gern_gen",
        .include = "-I /home/manya/gern/prez/library/array/impl",
    };
    runner.compile(options);
    return runner;
}
