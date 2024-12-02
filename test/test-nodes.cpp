
#include <gtest/gtest.h>
#include <iostream>

#include "annotations/data_dependency_language.h"
#include "test-utils.h"

#include "utils/error.h"

using namespace gern;

TEST(ExprNode, Literal) {
    Expr e1{(uint64_t)4};
    auto node = e1.ptr;
    ASSERT_EQ(node->getDatatype().getKind(), Datatype::Kind::UInt64);

    Expr e2{(uint32_t)4};
    node = e2.ptr;
    ASSERT_EQ(node->getDatatype().getKind(), Datatype::Kind::UInt32);

    Expr e3{(double)4};
    node = e3.ptr;
    ASSERT_EQ(node->getDatatype().getKind(), Datatype::Kind::Float64);
}

TEST(Expr, BinaryNodes) {
    Expr e = 4;
    Variable v{"v"};

    ASSERT_TRUE(getStrippedString((v + 4) / (v * 4)) == "((v+4)/(v*4))");
    ASSERT_TRUE(getStrippedString(v + 4 && v * 4) == "((v+4)&&(v*4))");
    ASSERT_TRUE(getStrippedString(v % 4 >= v * 4) == "((v%4)>=(v*4))");
}

TEST(StmtNode, Constraint) {
    Variable v("v_");
    Expr e = 1 - v;

    Constraint test(e == e);
    ASSERT_TRUE(getStrippedString(test) == "((1-v_)==(1-v_))");

    auto testDS = std::make_shared<const dummy::TestDataStructure>();
    Subset subset{testDS, {1 - v, v * 2}};

    ASSERT_TRUE(getStrippedString(subset) == "test{(1-v_),(v_*2)}");
}

TEST(Annotations, ConstrainPatterns) {
    Variable v("v");
    Variable v1("v1");
    auto testDS = std::make_shared<const dummy::TestDataStructure>();
    Subset s{testDS, {1 - v, v * 2}};

    ASSERT_NO_THROW(For(v, 0, 0, 0, Computes(Produces(s), Consumes(Subsets(s))))
                        .where(v == 1));
    ASSERT_THROW(For(v, 0, 0, 0, Computes(Produces(s), Consumes(Subsets(s))))
                     .where(v1 == 1),
                 error::UserError);
}
