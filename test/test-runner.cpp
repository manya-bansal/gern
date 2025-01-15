#include "annotations/data_dependency_language.h"
#include "library/array/annot/cpu-array.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"
#include "utils/error.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(Runner, FailGracefully) {

    auto inputDS = AbstractDataTypePtr(new const annot::MatrixCPU("input_con"));
    auto outputDS = AbstractDataTypePtr(new const annot::MatrixCPU("output_con"));

    annot::MatrixAddCPU add;
    std::vector<Compose> c = {add(inputDS, outputDS)};

    // Make sure Gern throws an exception and crash, and not just explode.

    Pipeline p(c);
    Runner run(p);

    // Try to compile with no include flag sets, so that the compilation
    // step fails.
    ASSERT_THROW(run.compile(Runner::Options()), error::UserError);
}