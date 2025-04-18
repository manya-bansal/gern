#include "codegen/lower.h"
#include "compose/composable.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(Stage, NotInScope) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPU("temp"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    // Stage is not in the current scope.
    ASSERT_THROW(Stage(tempDS, add_1(inputDS, outputDS)), error::UserError);
}

TEST(Stage, QueryInnerTile) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    // Useless stage, but should work.
    Composable call =
        Tile(outputDS["size"], v)(
            Stage(inputDS,
                  Tile(outputDS["size"], v)(
                      add_1(inputDS, outputDS))));

    Runner run(call);
    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(8);
    a.ascending();
    impl::ArrayCPU b(8);
    b.vvals(0.0f);
    int64_t v_val = 2;

    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {"v", &v_val},
    });

    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    a.destroy();
    b.destroy();
}

TEST(Stage, DoubleInnerTile) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    // Useless stage, but should work.
    Composable call =
        Tile(outputDS["size"], v)(
            Stage(inputDS,
                  Tile(outputDS["size"], v)(
                      Tile(outputDS["size"], v)(
                          add_1(inputDS, outputDS)))));

    Runner run(call);
    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(8);
    a.ascending();
    impl::ArrayCPU b(8);
    b.vvals(0.0f);
    int64_t v_val = 2;

    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {"v", &v_val},
    });

    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    a.destroy();
    b.destroy();
}

TEST(Stage, DoubleStage) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    // Useless stage, but should work.
    Composable call =
        Tile(outputDS["size"], v)(
            Stage(inputDS,
                  Tile(outputDS["size"], v)(
                      Stage(inputDS,
                            Tile(outputDS["size"], v)(
                                Tile(outputDS["size"], v)(
                                    add_1(inputDS, outputDS)))))));

    Runner run(call);
    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(16);
    a.ascending();
    impl::ArrayCPU b(16);
    b.vvals(0.0f);
    int64_t v_val = 2;

    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {"v", &v_val},
    });

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    a.destroy();
    b.destroy();
}

TEST(Stage, StageReduction) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));

    annot::reduction reduction;
    Variable v("v");
    Variable v1("v1");

    // Useless stage, but should work.
    Composable call =
        Tile(outputDS["size"], v)(
            Stage(outputDS,
                  Reduce(v1, v)(
                      Stage(inputDS,
                            reduction(inputDS, outputDS, v1)))));

    Runner run(call);
    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(16);
    a.ascending();
    impl::ArrayCPU b(16);
    b.vvals(0.0f);
    int64_t v_val = 2;
    int64_t v1_val = 16;

    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {"v", &v_val},
        {"v1", &v1_val},
    });

    for (int i = 0; i < 16; i++) {
        ASSERT_TRUE(b.data[i] == 8 * 15);
    }

    a.destroy();
    b.destroy();
}

TEST(Stage, StageInterface) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPU("output_con"));
    auto stageDS = annot::ArrayCPU("stage_con");

    annot::add_1 add_1;
    Variable v("v");
    Variable v1("v1");

    // Useless stage, but should work.
    Composable call =
        Tile(outputDS["size"], v)(
            Stage(outputDS, stageDS.newStageFunction(), stageDS.getNewInsertFunction(),
                  Tile(outputDS["size"], v)(
                      Tile(outputDS["size"], v)(
                          Stage(inputDS, stageDS.newStageFunction(),
                                Stage(outputDS, stageDS.newStageFunction(), stageDS.getNewInsertFunction(),
                                      add_1(inputDS, outputDS)))))));

    Runner run(call);
    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU a(8);
    a.ascending();
    impl::ArrayCPU b(8);
    b.vvals(0.0f);
    int64_t v_val = 2;

    run.evaluate({
        {inputDS.getName(), &a},
        {outputDS.getName(), &b},
        {"v", &v_val},
    });

    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(b.data[i] == (a.data[i] + 1));
    }

    a.destroy();
    b.destroy();
}
