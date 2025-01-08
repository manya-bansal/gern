
#include "annotations/data_dependency_language.h"
#include "library/array/annot/cpu-array.h"
#include "test-utils.h"
#include "utils/error.h"

#include <gtest/gtest.h>
#include <iostream>

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

    ASSERT_TRUE(test::getStrippedString((v + 4) / (v * 4)) == "((v+4)/(v*4))");
    ASSERT_TRUE(test::getStrippedString(v + 4 && v * 4) == "((v+4)&&(v*4))");
    ASSERT_TRUE(test::getStrippedString(v % 4 >= v * 4) == "((v%4)>=(v*4))");
}

TEST(StmtNode, Constraint) {
    Variable v("v_");
    Expr e = 1 - v;

    Constraint test(e == e);
    ASSERT_TRUE(test::getStrippedString(test) == "((1-v_)==(1-v_))");

    auto TestDSCPU = std::make_shared<const annot::ArrayCPU>();
    Subset subset{TestDSCPU, {1 - v, v * 2}};

    ASSERT_TRUE(test::getStrippedString(subset) == "test{(1-v_),(v_*2)}");
}

TEST(Annotations, ConstrainPatterns) {
    Variable v("v");
    Variable v1("v1");
    auto TestDSCPU = std::make_shared<const annot::ArrayCPU>();
    Subset s{TestDSCPU, {1 - v, v * 2}};
    ProducesSubset p_s{TestDSCPU, {v, v}};

    ASSERT_NO_THROW(For(v = Expr(0), Expr(0), Expr(0),
                        Computes(Produces(p_s), Consumes(Subsets(s))))
                        .where(v == 1));
    ASSERT_THROW(For(v = Expr(0), Expr(0), Expr(0), Computes(Produces(p_s), Consumes(Subsets(s))))
                     .where(v1 == 1),
                 error::UserError);
    ASSERT_NO_THROW(s.where(v == 1));
    ASSERT_THROW(s.where(v1 == 1), error::UserError);
}
