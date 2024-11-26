
#include <gtest/gtest.h>
#include <iostream>

#include "annotations/data_dependency_language.h"

using namespace gern;

TEST(StmtNode, Constraint) {
  Variable v("v");
  Expr e = 1 - v;
  std::cout << (e || e) << std::endl;
  Constraint test(v, e == e);
  std::cout << test << std::endl;
}