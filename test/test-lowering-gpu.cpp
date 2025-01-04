#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/pipeline.h"
#include "functions/elementwise.h"
#include "functions/reduction.h"
#include "library/array/gpu-array-lib.h"

#include "config.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(LoweringGPU, SingleElemFunction) {
    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::addGPU add_f;
    Variable v("v");

    std::vector<Compose> c = {add_f[{{"end", v}}](inputDS, outputDS)};
    Pipeline p(c);

    p.at_device().lower();
    // p.lower();
    // p.compile("-std=c++11 -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/");

    // // Now, actually run the function.
    // lib::TestArray a(10);
    // a.vvals(2.0f);
    // lib::TestArray b(10);
    // b.vvals(3.0f);
    // int64_t var = 10;

    // ASSERT_NO_THROW(p.evaluate({
    //     {inputDS->getName(), &a},
    //     {outputDS->getName(), &b},
    //     {v.getName(), &var},
    // }));

    // // Make sure we got the correct answer.
    // for (int i = 0; i < 10; i++) {
    //     ASSERT_TRUE(b.data[i] == 5.0f);
    // }

    // // Try running with insufficient number
    // // of arguments.
    // ASSERT_THROW(p.evaluate({
    //                  {inputDS->getName(), &a},
    //                  {outputDS->getName(), &b},
    //              }),
    //              error::UserError);

    // auto dummyDS = std::make_shared<const dummy::TestDSCPU>("dummy_ds");
    // // Try running the correct number of arguments,
    // // but with the wrong reference data-structure.
    // ASSERT_THROW(p.evaluate({
    //                  {inputDS->getName(), &a},
    //                  {dummyDS->getName(), &b},
    //                  {v.getName(), &var},
    //              }),
    //              error::UserError);

    // a.destroy();
    // b.destroy();
}
