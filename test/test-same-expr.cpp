#include "annotations/expr.h"
#include "library/array/annot/cpu-array.h"

#include <gtest/gtest.h>
#include <iostream>

using namespace gern;

TEST(SameExpr, Variable) {
    Variable a("a");
    Variable b("b");

    ASSERT_TRUE(isSameExpr(a, a));
    ASSERT_FALSE(isSameExpr(a, b));
    ASSERT_FALSE(isSameExpr(a, 0));
    ASSERT_FALSE(isSameExpr(a, Expr()));
    ASSERT_FALSE(isSameExpr(Expr(), a));
    ASSERT_TRUE(isSameExpr(Expr(), Expr()));
}

TEST(SameExpr, ADTMember) {
    AbstractDataTypePtr a = annot::ArrayCPU("a");
    AbstractDataTypePtr b = annot::ArrayCPU("b");

    ASSERT_TRUE(isSameExpr(a["a"], a["a"]));
    ASSERT_FALSE(isSameExpr(a["a"], b["a"]));
    ASSERT_FALSE(isSameExpr(a["a"], a["b"]));
}

TEST(SameExpr, GridDim) {
    Grid::Dim a(Grid::Dim::BLOCK_DIM_X);
    Grid::Dim a_copy(Grid::Dim::BLOCK_DIM_X);
    Grid::Dim b(Grid::Dim::BLOCK_DIM_Y);

    ASSERT_TRUE(isSameExpr(a, a));
    ASSERT_TRUE(isSameExpr(a, a_copy));
    ASSERT_FALSE(isSameExpr(a, b));
}

TEST(SameExpr, BinaryExpr) {
    Variable a("a");
    Variable b("b");
    Variable c("c");

    ASSERT_TRUE(isSameExpr(a + b, a + b));
    ASSERT_TRUE(isSameExpr(a - b, a - b));
    ASSERT_TRUE(isSameExpr(a * b, a * b));
    ASSERT_TRUE(isSameExpr(a / b, a / b));
    ASSERT_TRUE(isSameExpr(a % b, a % b));

    ASSERT_FALSE(isSameExpr(a + b, a + c));
    ASSERT_FALSE(isSameExpr(a, a + c));
    ASSERT_FALSE(isSameExpr(a + b, a));
    ASSERT_FALSE(isSameExpr(a + b, a - b));
}

TEST(SameExpr, CompoundExpr) {
    Variable a("a");
    Variable b("b");
    Variable c("c");

    ASSERT_TRUE(isSameExpr(a + b * c, a + b * c));
    ASSERT_TRUE(isSameExpr(a % b / c, a % b / c));

    ASSERT_FALSE(isSameExpr(a + b * c, a - b * c));
    ASSERT_FALSE(isSameExpr(a + b * c, a - b));
    ASSERT_FALSE(isSameExpr(a + b * c, a));
}