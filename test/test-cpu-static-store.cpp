#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/cpu-array-template.h"
#include "library/array/impl/cpu-array-template.h"
#include "library/matrix/annot/cpu-matrix-template.h"
#include "library/matrix/impl/cpu-matrix-template.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(StaticStore, Single) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("output_con"));

    annot::addStaticStore add_1;
    Variable v("v");
    Variable step("step");

    Composable program =
        Tile(outputDS["size"], step.bind(5))(
            add_1(inputDS, outputDS));

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

TEST(StaticStore, Multi) {
    auto inputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("input_con"));
    auto tempDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("temp", true));
    auto outputDS = AbstractDataTypePtr(new const annot::ArrayCPUTemplate<10>("output_con"));

    annot::addStaticStore add_1;
    Variable v("v");
    Variable step("step");

    Composable program = Composable(
        Tile(outputDS["size"], step.bind(5))(
            add_1(inputDS, tempDS),
            add_1(tempDS, outputDS)));

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
        ASSERT_TRUE(output.data[i] == (input.data[i] + 2));
    }
}

TEST(StaticStore, Matrix) {
	auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU("output_con"));

	int64_t row_val = 10;
    int64_t col_val = 5;

    annot::MatrixAddCPU add_1;
    Variable v("v");
    Variable l_x("l_x");
    Variable l_y("l_y");

    Composable program =
        Tile(outputDS["row"], l_x.bind(5))(
			Tile(outputDS["col"], l_y.bind(5))(
            	add_1(inputDS, outputDS)
			)
		);

    Runner run(program);

    ASSERT_NO_THROW(run.compile(test::cpuRunner("matrix")));

    impl::MatrixCPUTemplate input(row_val, col_val);
    input.ascending();
    impl::MatrixCPUTemplate output(row_val, col_val);

    ASSERT_NO_THROW(run.evaluate({
        {inputDS.getName(), &input},
        {outputDS.getName(), &output},
    }));

    // Make sure we got the correct answer.
    for (int i = 0; i < row_val * col_val; i++) {
        ASSERT_TRUE(output.data[i] == (input.data[i] + 1));
    }
}