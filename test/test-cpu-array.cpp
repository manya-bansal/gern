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

TEST(LoweringCPU, SingleElemFunction) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");

    annot::add add_f;
    Variable v("v");
    Variable step("step");

    std::vector<Compose> c = {add_f[{
        {"end", v},
        {"step", step},
    }](inputDS, outputDS)};
    Pipeline p(c);
    Runner run(p);

    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU b(10);
    b.vvals(3.0f);
    int64_t var = 10;
    int64_t step_val = 1;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {v.getName(), &var},
        {step.getName(), &step_val},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == 5.0f);
    }

    // Try running with insufficient number
    // of arguments.
    ASSERT_THROW(run.evaluate({
                     {inputDS->getName(), &a},
                     {outputDS->getName(), &b},
                     {v.getName(), &var},
                 }),
                 error::UserError);

    auto dummyDS = std::make_shared<const annot::ArrayCPU>("dummy_ds");
    // Try running the correct number of arguments,
    // but with the wrong reference data-structure.
    ASSERT_THROW(run.evaluate({
                     {inputDS->getName(), &a},
                     {dummyDS->getName(), &b},
                     {v.getName(), &var},
                     {step.getName(), &step_val},
                 }),
                 error::UserError);

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, SingleReduceFunction) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");

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
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
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
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
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
                     {inputDS->getName(), &a},
                     {outputDS->getName(), &b},
                 }),
                 error::UserError);

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MultiFunc) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input");
    auto tempDS = std::make_shared<const annot::ArrayCPU>("temp");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output");

    annot::add add_f;
    Variable v("v");
    Variable step("step");

    Pipeline p({add_f(inputDS, tempDS),
                add_f[{
                    {"end", v},
                    {"step", step},
                }](tempDS, outputDS)});

    Runner run(p);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU c(10);
    c.vvals(4.0f);
    int64_t var1 = 10;
    int64_t var2 = 1;

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &c},
        {v.getName(), &var2},
        {step.getName(), &var1},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(c.data[i] == 6.0f);
    }
}

TEST(LoweringCPU, SingleElemFunctionTemplated) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");

    annot::addTemplate add_f;
    Variable v("v");
    Variable step("step");

    std::vector<Compose> c = {add_f[{
        {"end", v},
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
    int64_t var = 10;
    int64_t step_val = 1;

    // Complain because the user is trying to set step.
    ASSERT_THROW(run2.evaluate({
                     {inputDS->getName(), &a},
                     {outputDS->getName(), &b},
                     {v.getName(), &var},
                     {step.getName(), &step_val},
                 }),
                 error::UserError);

    // No problem now.
    ASSERT_NO_THROW(run2.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {v.getName(), &var},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == 5.0f);
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MultiFunctionTemplated) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input");
    auto tempDS = std::make_shared<const annot::ArrayCPU>("temp");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output");

    annot::addTemplate add_f;
    Variable v("v");
    Variable step("step");

    Pipeline p({add_f(inputDS, tempDS),
                add_f[{
                    {"end", v},
                    {"step", step.bindToInt64(10)},
                }](tempDS, outputDS)});

    Runner run(p);
    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU c(10);
    c.vvals(4.0f);
    int64_t var2 = 1;

    // Temp should not be included.
    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &c},
        {v.getName(), &var2},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(c.data[i] == 6.0f);
    }
}

TEST(LoweringCPU, OverspecifiedGrid) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto tempDS = std::make_shared<const annot::ArrayCPU>("temp");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");

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
