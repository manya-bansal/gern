#include "annotations/abstract_function.h"
#include "compose/compose.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(FunctionPtr, Basic) {
    annot::add_1 add_1;
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    // Conctruct a simple gern program (no fusion).
    Composable gern_function(add_1(inputDS, outputDS));
    // Generate a gern function pointer from the gern program.
    FunctionPtr function_ptr(gern_function, Runner::Options());

    // Now construct a new gern program that uses the function pointer.
    Composable program(
        function_ptr(inputDS, outputDS));

    // Compile this program.
    Runner run(program);
    run.compile(test::cpuRunner("array"));
    // // Print the function pointer.
    // std::cout << function_ptr.getAnnotation() << std::endl;

    // std::cout << function_ptr.getFunction() << std::endl;
    // std::cout << function_ptr.getHeader()[0] << std::endl;
}
