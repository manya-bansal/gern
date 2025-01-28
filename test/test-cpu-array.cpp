#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
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

TEST(LoweringCPU, SingleReduceFunction) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    Variable v1("v1");
    Variable v2("v2");

    annot::reduction reduce_f;

    std::vector<Compose> c = {reduce_f[{
        {"end", v1},
        {"step", v2},
    }](inputDS, outputDS)};

    Pipeline p(c);
    Runner run(p);

    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU b(10);
    b.vvals(0.0f);
    int64_t var1 = 10;
    int64_t var2 = 1;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v2.getName(), &var2},
        {v1.getName(), &var1},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == 2.0f * 10);
    }

    b.vvals(0.0f);
    // Run again, we should be able to
    // repeatedly used the compiled pipeline.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v2.getName(), &var2},
        {v1.getName(), &var1},
    }));
    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == 2.0f * 10);
    }

    // Try running with insufficient number
    // of arguments.
    ASSERT_THROW(run.evaluate({
                     {inputDS.getName(), &a},
                     {outputDS.getName(), &b},
                 }),
                 error::UserError);

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MultiFunc) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add add_f;
    Variable v("v");
    Variable step("step");

    Pipeline p({add_f(inputDS, tempDS),
                add_f[{
                    {"step", step},
                }](tempDS, outputDS)});

    Runner run(p);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU c(10);
    c.vvals(4.0f);
    int64_t var1 = 10;

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &c},
        {step.getName(), &var1},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(c.data[i] == 6.0f);
    }

    a.destroy();
    c.destroy();
}

TEST(LoweringCPU, SingleElemFunctionTemplated) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::addTemplate add_f;
    Variable v("v");
    Variable step("step");

    std::vector<Compose> c = {add_f[{
        {"step", step},
    }](inputDS, outputDS)};

    Pipeline p(c);
    Runner run(p);

    // Complain since the user has not bound the step parameter to a concrete value.
    ASSERT_THROW(run.compile(test::cpuRunner("array")), error::UserError);

    c = {add_f[{
        {"end", v},
        {"step", step.bindToInt64(2)},
    }](inputDS, outputDS)};

    Pipeline p2(c);
    Runner run2(p2);

    // We bound the template parameter to a an integer value, all should be O.K..
    ASSERT_NO_THROW(run2.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU b(10);
    b.vvals(3.0f);
    int64_t step_val = 1;

    // Complain because the user is trying to set step.
    ASSERT_THROW(run2.evaluate({
                     {inputDS.getName(), &a},
                     {outputDS.getName(), &b},
                     {step.getName(), &step_val},
                 }),
                 error::UserError);

    // No problem now.
    ASSERT_NO_THROW(run2.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == 5.0f);
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MultiFunctionTemplated) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::addTemplate add_f;
    Variable v("v");
    Variable step("step");

    Pipeline p({add_f(inputDS, tempDS),
                add_f[{
                    {"step", step.bindToInt64(10)},
                }](tempDS, outputDS)});

    Runner run(p);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU c(10);
    c.vvals(4.0f);

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &c},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(c.data[i] == 6.0f);
    }

    a.destroy();
    c.destroy();
}

TEST(LoweringCPU, OverspecifiedGrid) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::addTemplate add_f;
    Variable v("v");
    Variable step("step");

    // Try binding step twice.
    // This is an overspecified pipeline.
    std::vector<Compose> c = {
        add_f[{
            {"step", step.bindToInt64(10)},
        }](inputDS, tempDS),
        add_f[{
            {"step", step.bindToInt64(10)},
        }](tempDS, outputDS),
    };

    Pipeline p(c);
    Runner run(p);

    ASSERT_THROW(run.compile(test::cpuRunner("array")), error::UserError);
}
