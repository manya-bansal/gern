#include "annotations/visitor.h"
#include "codegen/runner.h"
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

TEST(LoweringGPU, SingleElemFunctionNoBind) {
    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::addGPU add_f;
    Variable v("v");

    // No interval variable is being mapped to the grid,
    // this implementation runs the entire computation on a
    // single thread.
    std::vector<Compose> c = {add_f[{{"end", v}}](inputDS, outputDS)};
    Pipeline p(c);

    p.at_device();
    Runner run(p);

    run.compile(Runner::Options{
        "nvcc",
        "test.cu",
        "/tmp",
        " -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/",
        "",
    });

    // Now, actually run the function.
    lib::TestArrayGPU a(10);
    a.vvals(2.0f);
    lib::TestArrayGPU b(10);
    b.vvals(3.0f);
    int64_t var = 10;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {v.getName(), &var},
    }));

    lib::TestArray result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(result.data[i] == 5.0f);
    }

    result.destroy();

    // Try running with insufficient number
    // of arguments.
    ASSERT_THROW(run.evaluate({
                     {inputDS->getName(), &a},
                     {outputDS->getName(), &b},
                 }),
                 error::UserError);

    auto dummyDS = std::make_shared<const dummy::TestDSCPU>("dummy_ds");
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

TEST(LoweringGPU, SingleElemFunctionBind) {
    auto inputDS = std::make_shared<const dummy::TestDSGPU>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDSGPU>("output_con");

    test::addGPU add_f;
    Variable v("v");
    Variable blk("blk");

    std::vector<Compose> c = {add_f[{
        {"end", v},
        {"x", blk.bindToGrid(Grid::Property::BLOCK_ID_X)},
    }](inputDS, outputDS)};

    Pipeline p(c);
    p.at_device();

    Runner run(p);

    run.compile(Runner::Options{
        "nvcc",
        "test.cu",
        "/tmp",
        " -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/",
        "",
    });

    // Now, actually run the function.
    lib::TestArrayGPU a(10);
    a.vvals(2.0f);
    lib::TestArrayGPU b(10);
    b.vvals(3.0f);
    int64_t var = 10;

    ASSERT_NO_THROW(run.evaluate({
        {inputDS->getName(), &a},
        {outputDS->getName(), &b},
        {v.getName(), &var},
    }));

    lib::TestArray result = b.get();
    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(result.data[i] == 5.0f);
    }

    result.destroy();
    a.destroy();
    b.destroy();
}

// TEST(LoweringGPU, SingleReduceNoBind) {
//     auto inputDS = std::make_shared<const dummy::TestDSCPU>("input_con");
//     auto outputDS = std::make_shared<const dummy::TestDSCPU>("output_con");

//     Variable v1("v1");
//     Variable v2("v2");

//     test::reductionGPU reduce_f;

//     std::vector<Compose> c = {reduce_f[{{"end", v1}, {"step", v2}}](inputDS, outputDS)};

//     Pipeline p(c);

//     p.lower();
//     p.compile("-std=c++11 -I " + std::string(GERN_ROOT_DIR) + "/test/library/array/");

//     lib::TestArray a(10);
//     a.vvals(2.0f);
//     lib::TestArray b(10);
//     b.vvals(0.0f);
//     int64_t var1 = 10;
//     int64_t var2 = 1;

//     ASSERT_NO_THROW(p.evaluate({
//         {inputDS->getName(), &a},
//         {outputDS->getName(), &b},
//         {v2.getName(), &var2},
//         {v1.getName(), &var1},
//     }));

//     // Make sure we got the correct answer.
//     for (int i = 0; i < 10; i++) {
//         ASSERT_TRUE(b.data[i] == 2.0f * 10);
//     }

//     b.vvals(0.0f);
//     // Run again, we should be able to
//     // repeatedly used the compiled pipeline.
//     ASSERT_NO_THROW(p.evaluate({
//         {inputDS->getName(), &a},
//         {outputDS->getName(), &b},
//         {v2.getName(), &var2},
//         {v1.getName(), &var1},
//     }));
//     // Make sure we got the correct answer.
//     for (int i = 0; i < 10; i++) {
//         ASSERT_TRUE(b.data[i] == 2.0f * 10);
//     }

//     // Try running with insufficient number
//     // of arguments.
//     ASSERT_THROW(p.evaluate({
//                      {inputDS->getName(), &a},
//                      {outputDS->getName(), &b},
//                  }),
//                  error::UserError);

//     a.destroy();
//     b.destroy();
// }
