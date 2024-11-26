
#include <gtest/gtest.h>
#include <iostream>

#include "annotations/data_dependency_language.h"

using namespace gern;

class TestDataStructure : public AbstractDataType {
public:
  std::string getName() const override { return "test"; }
};

TEST(StmtNode, Constraint) {
  Variable v("v");
  Expr e = 1 - v;
  std::cout << (e || e) << std::endl;
  Constraint test(v, e == e);
  std::cout << test << std::endl;

  auto testDataSTructure = std::make_shared<const TestDataStructure>();
  Subset subset{testDataSTructure, {1 - v, v * 2}};
  std::cout << subset << std::endl;
}