
#include <gtest/gtest.h>
#include <iostream>

#include "utils/error.h"

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

  auto testDataStructure = std::make_shared<const dummy::TestDataStructure>();
  Subset s{testDataStructure, {2, 4, 5}};
  Variable v{"v"};

  ASSERT_FALSE(isValidDataDependencyPattern(
      Computes(Produces(s), Consumes(Produces(s)))));
  ASSERT_FALSE(isValidDataDependencyPattern(For(v, 0, 0, 0, Produces(s))));
  ASSERT_FALSE(isValidDataDependencyPattern(
      For(v, 0, 0, 0, (For(v, 0, 0, 0, Computes(Produces(s), Consumes(s)))))));
}

TEST(Annotations, ConstraintedStmts) {

  Variable v{"v"};
  auto testDS = std::make_shared<const dummy::TestDataStructure>();
  Subset s{testDS, {1, 4, 5}};

  ASSERT_THROW(Computes(Produces(s), Consumes(s)).where(v == 1),
               error::UserError);

  // Cannot tag the consumer statement since the consumer statement does not use
  // v.
  ASSERT_THROW(
      For(v, 0, 0, 0, Computes(Produces(s), Consumes(s.where(v == 1)))),
      error::UserError);

  // Can tag at the level of the for loop
  ASSERT_NO_THROW(
      For(v, 0, 0, 0, Computes(Produces(s), Consumes(s))).where(v == 1));

  // Now we can tag the consumer since its subset description uses v.
  Subset sv{testDS, {v, 4, 5}};
  ASSERT_NO_THROW(
      For(v, 0, 0, 0, Computes(Produces(s), Consumes(sv.where(v == 1)))));
}