#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/cpu-array-template.h"
#include "library/array/impl/cpu-array-template.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(StaticStore, Single) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("output_con"));

    annot::addStaticStore add_f;
    Variable v("v");
    Variable step("step");

    Composable program =
        Tile(outputDS["size"], step.bindToInt64(5))(
            add_f(inputDS, outputDS));

    Runner run(program);

    ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

    impl::ArrayCPUTemplate<10> input;
    input.ascending();
    impl::ArrayCPUTemplate<10> output;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &input},
        {outputDS.getName(), &output},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(output.data[i] == (input.data[i] + 1));
    }
}

// TEST(StaticStore, Multi) {
//     auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("input_con"));
//     auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("temp", true));
//     auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("output_con"));

//     annot::addStaticStore add_f;
//     Variable v("v");
//     Variable step("step");

//     std::vector<Compose> c = {
//         add_f(inputDS, tempDS),
//         add_f[{
//             {"step", step.bindToInt64(5)},
//         }](tempDS, outputDS)};

//     Pipeline p(c);
//     Runner run(p);

//     ASSERT_NO_THROW(run.compile(test::cpuRunner("array")));

//     impl::ArrayCPUTemplate<10> input;
//     input.vvals(2.0f);
//     impl::ArrayCPUTemplate<10> output;
//     output.vvals(4.0f);

//     ASSERT_NO_THROW(run.evaluate({
//         {inputDS.getName(), &input},
//         {outputDS.getName(), &output},
//     }));

//     // Make sure we got the correct answer.
//     for (int i = 0; i < 10; i++) {
//         ASSERT_TRUE(output.data[i] == 6.0f);
//     }
// }