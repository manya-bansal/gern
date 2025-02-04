#include "codegen/lower.h"
#include "compose/composable.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(ComposableTest, NoFusion) {

    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    Composable call =
        Composable({
            add_1(inputDS, tempDS),
            add_1(tempDS, outputDS),
        });

    std::cout << call.getAnnotation() << std::endl;

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);
    b.vvals(0.0f);

    Runner run(call);

    run.compile(test::cpuRunner("array"));
    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    });

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
    }

    a.destroy();
    b.destroy();
}

TEST(ComposableTest, NestedFusion) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1 add_1;
    Variable v("x");
    Variable v1("y");

    Composable program =
        Tile(outputDS["size"], v)(
            Tile(outputDS["size"], v1)(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS)));

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);
    b.vvals(0.0f);
    int64_t step_1 = 5;
    int64_t step_2 = 1;

    Runner run(program);
    run.compile(test::cpuRunner("array"));
    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &step_1},
        {v1.getName(), &step_2},
    });

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
    }

    std::cout << "?" << std::endl;

    a.destroy();
    std::cout << "???" << std::endl;

    b.destroy();
    std::cout << "????????" << std::endl;
}

TEST(ComposableTest, FusionSameScope) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    Composable program({
        Tile(tempDS["size"], v)(
            add_1(inputDS, tempDS)),
        Tile(outputDS["size"], v1)(
            add_1(tempDS, outputDS)),
    });

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    int64_t step_1 = 5;
    int64_t step_2 = 1;

    Runner run(program);
    run.compile(test::cpuRunner("array"));

    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &step_1},
        {v1.getName(), &step_2},
    });

    //  Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
    }

    a.destroy();
    b.destroy();
}