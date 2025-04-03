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

TEST(LoweringCPU, OrderedParameters) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add_1 add_1;
    Variable v("v");
    Variable step("step");

    Composable program(
        add_1(inputDS, outputDS));

    auto badDS = AbstractDataTypePtr(new const annot::ArrayCPU("bad"));
    auto bad_init = [&] {
        Runner bad_run = {program, std::vector<Parameter>{Parameter{outputDS}, Parameter{badDS}}};
        bad_run.compile(test::cpuRunner("array"));
    };
    ASSERT_THROW(
        bad_init(),
        error::UserError);

    Runner run(program, std::vector<Parameter>{Parameter{outputDS}, Parameter{inputDS}});

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
