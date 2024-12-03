#include "functions/elementwise.h"
#include <gtest/gtest.h>

#include "test-utils.h"
using namespace gern;

TEST(Functions, ElementWise) {
    auto inputDS = std::make_shared<const dummy::TestDS>("input");
    auto outputDS = std::make_shared<const dummy::TestDS>("output");

    test::add add_f;
    add_f(inputDS, outputDS);
}