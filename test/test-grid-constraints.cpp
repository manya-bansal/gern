#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/gpu-array.h"
#include "test-gpu-utils.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(GridConstraints, Simple) {
    std::cout << std::is_base_of_v<Constraint, gern::Eq> << std::endl;
    std::cout << "Hi" << std::endl;
}