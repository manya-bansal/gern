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

TEST(LoweringCPU, ReductionNoTile) {

    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::reduction reduction;

    Composable program(
        reduction(inputDS, outputDS));

    Runner run(program);
    run.compile(test::cpuRunner("array"));

    int64_t size = 10;
    impl::ArrayCPU a(size);
    a.ascending();
    impl::ArrayCPU b(size);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2)));
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, ReductionOuterTile) {

    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::reduction reduction;
    Variable v("v");

    Composable program(
        Tile(outputDS["size"], v)(
            reduction(inputDS, outputDS)));

    Runner run(program);
    run.compile(test::cpuRunner("array"));

    int64_t size = 10;
    int64_t v_tile = 1;
    impl::ArrayCPU a(size);
    a.ascending();
    impl::ArrayCPU b(size);
    b.vvals(0.0f);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2)));
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, ReductionInnerTile) {

    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::reduction reduction;
    Variable v("v");

    Composable program(
        Tile(outputDS["size"], v)(
            Reduce(outputDS["size"], v)(
                reduction(inputDS, outputDS))));

    Runner run(program);
    run.compile(test::cpuRunner("array"));

    int64_t size = 10;
    int64_t v_tile = 1;
    impl::ArrayCPU a(size);
    a.ascending();
    impl::ArrayCPU b(size);
    b.vvals(0.0f);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2)));
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, TileReductions) {

    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::reduction reduction;
    Variable v("v");
    Variable v1("v1");

    Composable program(
        Tile(outputDS["size"], v)(
            Tile(outputDS["size"], v1)(
                Reduce(outputDS["size"], v)(
                    Reduce(outputDS["size"], v1)(
                        reduction(inputDS, outputDS))))));

    Runner run(program);
    run.compile(test::cpuRunner("array"));

    int64_t size = 10;
    int64_t v_tile = 5;
    int64_t v1_tile = 5;
    impl::ArrayCPU a(size);
    a.ascending();
    impl::ArrayCPU b(size);
    b.vvals(0.0f);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
        {v1.getName(), &v1_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2)));
    }

    a.destroy();
    b.destroy();
}

TEST(LoweringCPU, MixReductionStrategy) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::reduction reduction;
    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    Composable program(
        Tile(outputDS["size"], v)(
            Tile(outputDS["size"], v1)(
                Reduce(outputDS["size"], v)(
                    Reduce(outputDS["size"], v1)(
                        add_1(inputDS, tempDS),
                        reduction(tempDS, outputDS))))));

    Runner run_1(program);
    run_1.compile(test::cpuRunner("array"));

    // Should be exactly the same as the unfused version.
    int64_t size = 10;
    int64_t v_tile = 5;
    int64_t v1_tile = 5;
    impl::ArrayCPU a(size);
    a.ascending();
    impl::ArrayCPU b(size);
    b.vvals(0.0f);

    ASSERT_NO_THROW(run_1.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
        {v1.getName(), &v1_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2) + size));
    }

    program =
        Tile(outputDS["size"], v)(
            Tile(outputDS["size"], v1)(
                add_1(inputDS, tempDS),
                Reduce(outputDS["size"], v)(
                    Reduce(outputDS["size"], v1)(
                        reduction(tempDS, outputDS)))));

    Runner run_2(program);
    run_2.compile(test::cpuRunner("array"));

    b.vvals(0.0f);
    ASSERT_NO_THROW(run_2.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
        {v1.getName(), &v1_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2) + size));
    }

    a.destroy();
    b.destroy();
}