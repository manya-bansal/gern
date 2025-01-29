
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
    ASSERT_TRUE(test::getStrippedString(Argument(v + v)) == "(v+v)");
}

TEST(StmtNode, Constraint) {
    Variable v("v_");
    Expr e = 1 - v;

    Constraint test(e == e);
    ASSERT_TRUE(test::getStrippedString(e == e) == "((1-v_)==(1-v_))");
    ASSERT_TRUE(test::getStrippedString(e <= e) == "((1-v_)<=(1-v_))");
    ASSERT_TRUE(test::getStrippedString(e != e) == "((1-v_)!=(1-v_))");
    ASSERT_TRUE(test::getStrippedString(e < e) == "((1-v_)<(1-v_))");
    ASSERT_TRUE(test::getStrippedString(e > e) == "((1-v_)>(1-v_))");
    ASSERT_TRUE(test::getStrippedString(e && e) == "((1-v_)&&(1-v_))");
    ASSERT_TRUE(test::getStrippedString(e || e) == "((1-v_)||(1-v_))");

    auto TestDSCPU = AbstractDataTypePtr(new const annot::ArrayCPU());
    SubsetObj subset{TestDSCPU, {1 - v, v * 2}};

    ASSERT_TRUE(test::getStrippedString(subset) == "test{(1-v_),(v_*2)}");
}

TEST(Annotations, ConstrainPatterns) {
    Variable v("v");
    Variable v1("v1");
    auto TestDSCPU = AbstractDataTypePtr(new const annot::ArrayCPU());
    SubsetObj s{TestDSCPU, {1 - v, v * 2}};

    ASSERT_NO_THROW(For(v = Expr(0), TestDSCPU["dummy"], v,
                        Computes(Produces::Subset(TestDSCPU, {v, v}),
                                 Consumes(SubsetObjMany(s))))
                        .where(v == 1));
    ASSERT_THROW(For(v = Expr(0), TestDSCPU["dummy"], v,
                     Computes(
                         Produces::Subset(TestDSCPU, {v, v}),
                         Consumes(SubsetObjMany(s))))
                     .where(v1 == 1),
                 error::UserError);
    ASSERT_NO_THROW(s.where(v == 1));
    ASSERT_THROW(s.where(v1 == 1), error::UserError);
}

TEST(Annotations, NullNode) {
    // Make sure Gern doesn't explode when
    // a null pointer is passed.
    Expr().str();
    Stmt().str();
    Constraint().str();
    AbstractDataTypePtr().str();
    ASSERT_THROW(AbstractDataTypePtr().getInsertFunction(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().getQueryFunction(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().getName(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().getType(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().getFields(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().getAllocateFunction(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().freeQuery(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().insertQuery(), error::InternalError);
    ASSERT_THROW(AbstractDataTypePtr().freeAlloc(), error::InternalError);
    Argument().str();
}
