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

TEST(LowerinCPU, ReductionNoTile) {

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
}

TEST(LowerinCPU, ReductionOuterTile) {

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

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2)));
    }
}

TEST(LowerinCPU, ReductionInnerTile) {

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

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {v.getName(), &v_tile},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(b.data[i] == ((size * (size - 1) / 2)));
    }
}