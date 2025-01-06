#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "compose/runner.h"

#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"

#include "config.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringCPU, SingleElemFunction) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");

    annot::add add_f;
    Variable v("v");

    std::vector<Compose> c = {add_f[{{"end", v}}](inputDS, outputDS)};
    Pipeline p(c);
    Runner run(p);

    run.compile(Runner::Options{
        .filename = "test",
        .prefix = "/tmp",
        .include = " -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/impl",
    });

    impl::ArrayCPU a(10);
    a.vvals(2.0f);
    impl::ArrayCPU b(10);
    b.vvals(3.0f);
    int64_t var = 10;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {v.getName(), &var},
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
                 }),
                 error::UserError);

    auto dummyDS = std::make_shared<const annot::ArrayCPU>("dummy_ds");
    // Try running the correct number of arguments,
    // but with the wrong reference data-structure.
    ASSERT_THROW(run.evaluate({
                     {inputDS->getName(), &a},
                     {dummyDS->getName(), &b},
                     {v.getName(), &var},
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

    std::vector<Compose> c = {reduce_f[{{"end", v1}, {"step", v2}}](inputDS, outputDS)};

    Pipeline p(c);
    Runner run(p);

    run.compile(Runner::Options{
        .filename = "test",
        .prefix = "/tmp",
        .include = " -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/impl",
    });

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

TEST(LoweringCPU, MultiFunction) {
    auto inputDS = std::make_shared<const annot::ArrayCPU>("input_con");
    auto outputDS = std::make_shared<const annot::ArrayCPU>("output_con");
    annot::add add_f;

    Compose compose{{add_f(inputDS, outputDS),
                     Compose(add_f(outputDS, outputDS))}};
    Pipeline p({compose});

    // Currently, only pipelines with one function call
    // can be lowered. This will (obviously) be removed
    // as I make progress!
    ASSERT_THROW(p.lower(), error::InternalError);
}