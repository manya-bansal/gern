#include "annotations/visitor.h"
#include "compose/compose.h"
#include "functions/elementwise.h"
#include "test-utils.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(Lowering, SingleFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input_con");
    auto outputDS = std::make_shared<const dummy::TestDS>("output_con");
    test::add add_f;
    Compose compose{add_f(inputDS, outputDS)};
    std::cout << compose << std::endl;
}