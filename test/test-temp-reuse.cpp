#include "codegen/lower.h"
#include "compose/composable.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"
#include "test-utils.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

static void run_and_check(Composable call, int size) {

    Runner run(call);
    run.compile(test::cpuRunner("array"));

    impl::ArrayCPU input(10);
    input.ascending();
    impl::ArrayCPU output(10);

    run.evaluate({
        {"input_con", &input},
        {"output_con", &output},
    });

    for (int i = 0; i < size; i++) {
        ASSERT_TRUE(output.data[i] == (input.data[i] + size));
    }

    input.destroy();
    output.destroy();
}

TEST(TempReuse, SingleTemp) {
    auto inputDS = annot::ArrayCPU("input_con");
    auto temp1DS = annot::ArrayCPU("temp1_con");
    auto temp2DS = annot::ArrayCPU("temp2_con");
    auto temp3DS = annot::ArrayCPU("temp3_con");
    auto outputDS = annot::ArrayCPU("output_con");

    annot::add_1 add_1;
    Variable v("v");

    Composable call({
        add_1(inputDS, temp1DS),
        add_1(temp1DS, temp2DS),
        add_1(temp2DS, temp3DS),
        add_1(temp3DS, outputDS),
    });

    run_and_check(call, 4);
}

TEST(TempReuse, TwoTemps) {
    auto inputDS = annot::ArrayCPU("input_con");
    auto temp1DS = annot::ArrayCPU("temp1_con");
    auto temp2DS = annot::ArrayCPU("temp2_con");
    auto temp3DS = annot::ArrayCPU("temp3_con");
    auto temp4DS = annot::ArrayCPU("temp4_con");
    auto outputDS = annot::ArrayCPU("output_con");

    annot::add_1 add_1;
    Variable v("v");

    Composable call({
        add_1(inputDS, temp1DS),
        add_1(temp1DS, temp2DS),
        add_1(temp2DS, temp3DS),
        add_1(temp3DS, temp4DS),
        add_1(temp4DS, outputDS),
    });

    run_and_check(call, 5);
}

TEST(TempReuse, ThreeTemps) {
    auto inputDS = annot::ArrayCPU("input_con");
    auto temp1DS = annot::ArrayCPU("temp1_con");
    auto temp2DS = annot::ArrayCPU("temp2_con");
    auto temp3DS = annot::ArrayCPU("temp3_con");
    auto temp4DS = annot::ArrayCPU("temp4_con");
    auto temp5DS = annot::ArrayCPU("temp5_con");
    auto outputDS = annot::ArrayCPU("output_con");

    annot::add_1 add_1;
    Variable v("v");

    Composable call({
        add_1(inputDS, temp1DS),
        add_1(temp1DS, temp2DS),
        add_1(temp2DS, temp3DS),
        add_1(temp3DS, temp4DS),
        add_1(temp4DS, temp5DS),
        add_1(temp5DS, outputDS),
    });

    run_and_check(call, 6);
}
