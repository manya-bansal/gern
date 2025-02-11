#include "annotations/abstract_function.h"
#include "compose/compose.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(FunctionPtr, Basic) {
    annot::add_1 add_1;
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    // Construct a simple gern program (no fusion).
    Composable gern_function(add_1(inputDS, outputDS));
    // Generate a gern function pointer from the gern program.
    Runner::Options options = test::cpuRunner("array");
    options.filename = "simple_func.cpp";
    FunctionPtr function_ptr(gern_function, options);

    // Now construct a new gern program that uses the function pointer.
    Composable program(
        function_ptr(inputDS, outputDS));

    // Compile this program.
    Runner run(program);
    run.compile(test::cpuRunner("array"));

    // Evaluate the program.
    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    run.evaluate({{"input_con", &a},
                  {"output_con", &b}});

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    a.destroy();
    b.destroy();
}

TEST(FunctionPtr, FuseInner) {
    annot::add_1 add_1;
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp_con"));

    Variable size("size");
    // Construct a simple gern program (with fusion).
    Composable gern_function(
        Tile(outputDS["size"], size)(
            add_1(inputDS, tempDS),
            add_1(tempDS, outputDS)));
    // Generate a gern function pointer from the gern program.
    Runner::Options options = test::cpuRunner("array");
    options.filename = "simple_func.cpp";
    FunctionPtr function_ptr(gern_function, options);

    // Now construct a new gern program that uses the function pointer.
    Composable program(
        function_ptr(inputDS, outputDS, size));

    // Compile this program.
    Runner run(program);
    run.compile(test::cpuRunner("array"));

    // Evaluate the program.
    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);
    int64_t size_val = 5;

    run.evaluate({{inputDS.getName(), &a},
                  {outputDS.getName(), &b},
                  {size.getName(), &size_val}});

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
    }

    a.destroy();
    b.destroy();
}

TEST(FunctionPtr, FuseBoth) {
    annot::add_1 add_1;
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp_con"));

    Variable size("size");
    // Construct a simple gern program.
    Composable gern_function(
        Tile(outputDS["size"], size)(
            add_1(inputDS, tempDS),
            add_1(tempDS, outputDS)));
    // Generate a gern function pointer from the gern program.
    Runner::Options options = test::cpuRunner("array");
    options.filename = "simple_func.cpp";
    FunctionPtr function_ptr(gern_function, options);

    // Now construct a new gern program that uses the function pointer.
    Composable program(
        Tile(outputDS["size"], size)(
            function_ptr(inputDS, tempDS, size),
            function_ptr(tempDS, outputDS, size)));

    // Compile this program.
    Runner run(program);
    run.compile(test::cpuRunner("array"));

    // Evaluate the program.
    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);
    int64_t size_val = 5;

    run.evaluate({{inputDS.getName(), &a},
                  {outputDS.getName(), &b},
                  {size.getName(), &size_val}});

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 4));
    }

    a.destroy();
    b.destroy();
}
