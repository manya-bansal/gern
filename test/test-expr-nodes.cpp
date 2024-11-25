
#include <gtest/gtest.h>
#include <iostream>

#include "annotations/data_dependency_language.h"

using namespace gern;

TEST(Expr, Literal) {
  Expr e1{(uint64_t)4};
  auto node = e1.getNode();
  ASSERT_EQ(node->getDatatype().getKind(), Datatype::Kind::UInt64);

  Expr e2{(uint32_t)4};
  node = e2.getNode();
  ASSERT_EQ(node->getDatatype().getKind(), Datatype::Kind::UInt32);

  Expr e3{(double)4};
  node = e3.getNode();
  ASSERT_EQ(node->getDatatype().getKind(), Datatype::Kind::Float64);
}
