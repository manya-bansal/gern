#include "codegen/lower.h"
#include "compose/composable.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

template<typename Input, typename Output, typename Operations>
Composable name(Input &&input, Output &&output, Operations operations) {
    // Create AbstractDataTypePtr from input and output
    // AbstractDataTypePtr inputDS = AbstractDataTypePtr(std::forward<Input>(input));
    // AbstractDataTypePtr outputDS = AbstractDataTypePtr(std::forward<Output>(output));

    return operations(input, output);
    // // Apply operations in the body
    // return Composable({
    //     (operations(inputDS, outputDS), ...)  // Apply each operation in the fold expression
    // });
}

TEST(NewInterfaceTest, Basic) {

    annot::add_1 add_1;

    auto call = name(annot::ArrayCPU("input_con"), annot::ArrayCPU("output_con"),
                     [&](annot::ArrayCPU input, annot::ArrayCPU output) {
                         return Composable({
                             add_1(input, output),
                         });
                     });

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    Runner run(call);

    run.compile(test::cpuRunner("array"));
    run.evaluate({
        {"input_con", &a},
        {"output_con", &b},
    });

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }
}

#define Fern(input_decl, output_decl, program) \
    Composable call;                           \
    {                                          \
        input_decl;                            \
        output_decl;                           \
        call = program;                        \
    }

TEST(NewInterfaceTest, Basic2) {
    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    annot::add_1 add_1;

    Fern(annot::ArrayCPU input("input_con"), annot::ArrayCPU output("output_con"),
         add_1(input, output));

    Runner run(call);

    run.compile(test::cpuRunner("array"));
    run.evaluate({
        {"input_con", &a},
        {"output_con", &b},
    });

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }
}