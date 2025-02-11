#include "annotations/visitor.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringCPU, OneFuncNoFusion) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add_1 add_1;
    Variable v("v");
    Variable step("step");

    Composable program(
        add_1(inputDS, outputDS));

    Runner run(program);

    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    // Try running with insufficient number
    // of arguments.
    ASSERT_THROW(run.evaluate({
                     {inputDS.getName(), &a},
                 }),
                 error::UserError);

    auto dummyDS = AbstractDataTypePtr(new const annot::ArrayCPU("dummy_ds"));
    // Try running the correct number of arguments,
    // but with the wrong reference data-structure.
    ASSERT_THROW(run.evaluate({
                     {inputDS.getName(), &a},
                     {dummyDS.getName(), &b},
                 }),
                 error::UserError);
    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MultiFunc) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1 add_1;
    Variable step("step");

    Composable program =
        Composable(
            Tile(outputDS["size"], step)(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS)));

    Runner run(program);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU c(10);
    int64_t var1 = 2;

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &c},
        {step.getName(), &var1},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(c.data[i] == (a.data[i] + 2));
    }

    a.destroy();
    c.destroy();
}

TEST(LoweringCPU, SingleElemFunctionTemplated) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add1Template add_1;
    Variable v("v");
    Variable step("step");

    Composable program =
        Composable(
            Tile(outputDS["size"], step)(
                add_1(inputDS, outputDS)));

    Runner run_error(program);

    // Complain since the user has not bound the step parameter to a concrete value.
    ASSERT_THROW(run_error.compile(test::cpuRunner("array")), error::UserError);

    program =
        Composable(
            Tile(outputDS["size"], step.bind(5))(
                add_1(inputDS, outputDS)));

    Runner run(program);

    // We bound the template parameter to a an integer value, all should be O.K..
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    int64_t step_val = 1;

    // Complain because the user is trying to set step.
    ASSERT_THROW(run.evaluate({
                     {inputDS.getName(), &a},
                     {outputDS.getName(), &b},
                     {step.getName(), &step_val},
                 }),
                 error::UserError);

    // No problem now.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MultiFunctionTemplated) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add1Template add_1;
    Variable v("v");
    Variable step("step");

    Composable program =
        Composable(
            Tile(outputDS["size"], step.bind(5))(
                add_1(inputDS, tempDS),
                add_1(tempDS, outputDS)));

    Runner run(program);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, OverspecifiedBinding) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add1Template add_1;

    Variable v("v");
    Variable step("step");

    auto add_1_specialized = &add_1[{{"step", step.bind(5)}}];
    // Try binding step twice.
    // This is an overspecified pipeline.
    Composable program =
        Composable(
            Tile(outputDS["size"], step.bind(5))(
                add_1_specialized->operator()(inputDS, tempDS),
                add_1(tempDS, outputDS)));

    Runner run(program);
    ASSERT_THROW(run.compile(test::cpuRunner("array")), error::UserError);
}

TEST(LoweringCPU, FunctionWithFloat) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1_float add_1_float;
    Variable v("v", Datatype::Float32);
    Variable step("step");

    Composable program =
        Composable(
            Tile(outputDS["size"], step.bind(5))(
                add_1_float(inputDS, tempDS, v),
                add_1_float(tempDS, outputDS, v)));

    Runner run(program);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.ascending();
    impl::ArrayCPU b(10);
    float v_val = 1.0;

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_val},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 2));
    }

    a.destroy();
    b.destroy();
}