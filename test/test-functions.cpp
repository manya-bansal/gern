#include "compose/compose.h"
#include "functions/elementwise.h"
#include "test-utils.h"
#include <gtest/gtest.h>

using namespace gern;

TEST(Functions, SingleFunction) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input");
    auto outputDS = std::make_shared<const dummy::TestDS>("output");

    test::add add_f;
    Compose compose{{add_f(inputDS, outputDS)}};
}