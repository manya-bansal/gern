
#include <gtest/gtest.h>
#include <iostream>

#include "annotations/data_dependency_language.h"
#include <iostream>

using namespace gern;

class TestDataStructure : public AbstractDataType {
public:
  std::string getName() const override { return "test"; }
};

TEST(ExprNode, Literal) {
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

template <typename T> static std::string getStrippedString(T e) {
  std::stringstream ss;
  ss << e;
  auto str = ss.str();
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  return std::string(str);
}

TEST(Expr, BinaryNodes) {
  Expr e = 4;
  Variable v{"v"};

  ASSERT_TRUE(getStrippedString(v + 4 / v * 4) == "v+4/v*4");
  ASSERT_TRUE(getStrippedString(v + 4 && v * 4) == "((v+4)&&(v*4))");
  ASSERT_TRUE(getStrippedString(v % 4 >= v * 4) == "((v%4)>=(v*4))");
}

TEST(StmtNode, Constraint) {
  Variable v("v_");
  Expr e = 1 - v;
  std::cout << (e || e) << std::endl;
  Constraint test(v, e == e);

  ASSERT_TRUE(getStrippedString(test) == "v_where((1-v_)==(1-v_))");

  auto testDataSTructure = std::make_shared<const TestDataStructure>();
  Subset subset{testDataSTructure, {1 - v, v * 2}};

  std::cout << getStrippedString(subset) << std::endl;
  ASSERT_TRUE(getStrippedString(subset) == "test{1-v_,v_*2}");
}