
#include <gtest/gtest.h>
#include <iostream>

#include "test-utils.h"

using namespace gern;

TEST(Annotations, ValidAnnotations) {

  auto testDataSTructure = std::make_shared<const dummy::TestDataStructure>();
  Subset s{testDataSTructure, {2, 4, 5}};
  Variable v{"v"};
  Variable v1{"v1"};

  ASSERT_TRUE(isValidDataDependencyPattern(Computes(Produces(s), Consumes(s))));
  ASSERT_TRUE(isValidDataDependencyPattern(
      For(v, 0, 0, 0, Computes(Produces(s), Consumes(s)))));
  ASSERT_TRUE(isValidDataDependencyPattern(
      For(v1, 0, 0, 0, (For(v, 0, 0, 0, Computes(Produces(s), Consumes(s)))))));
}

TEST(Annotations, notValidAnnotations) {

  auto testDataSTructure = std::make_shared<const dummy::TestDataStructure>();
  Subset s{testDataSTructure, {2, 4, 5}};
  Variable v{"v"};

  ASSERT_FALSE(isValidDataDependencyPattern(
      Computes(Produces(s), Consumes(Produces(s)))));
  ASSERT_FALSE(isValidDataDependencyPattern(
      For(v, 0, 0, 0, Produces(s))));
  ASSERT_FALSE(isValidDataDependencyPattern(
      For(v, 0, 0, 0, (For(v, 0, 0, 0, Computes(Produces(s), Consumes(s)))))));
}