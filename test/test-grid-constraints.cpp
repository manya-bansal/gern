#include "annotations/visitor.h"
#include "compose/compose.h"
#include "compose/runner.h"
#include "config.h"
#include "library/array/annot/gpu-array.h"
#include "test-gpu-utils.h"
#include "test-utils.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace gern;

TEST(GridConstraints, ConstrainDimension) {
    Variable v("v");
    Variable v1("v1");
    auto TestDSCPU = AbstractDataTypePtr(new const annot::ArrayCPU());
    SubsetObj s{TestDSCPU, {1 - v, v * 2}};

    // Try to constrain a dimension that does not belong to this
    // function.
    ASSERT_THROW(Tileable(v = Expr(0), TestDSCPU["dummy"], v,
                          Computes(
                              Produces::Subset(TestDSCPU, {v, v}),
                              Consumes(SubsetObjMany(s))))
                     .assumes(Grid::Dim::BLOCK_DIM_X == v),
                 error::UserError);

    // Catch if we try to constrain a dimension at the same level.
    // Try to constrain a dimension that does belong to this
    // function.
    ASSERT_THROW(Tileable(v = Expr(0), TestDSCPU["dummy"], v,
                          Computes(
                              Produces::Subset(TestDSCPU, {v, v}),
                              Consumes(SubsetObjMany(s))))
                     .occupies({Grid::Unit::THREAD_X})
                     .assumes(Grid::Dim::BLOCK_DIM_Y == v),
                 error::UserError);

    // Complain if annotation tries to constrain a parent.
    ASSERT_THROW(Tileable(v = Expr(0), TestDSCPU["dummy"], v,
                          Computes(
                              Produces::Subset(TestDSCPU, {v, v}),
                              Consumes(SubsetObjMany(s))))
                     .occupies({Grid::Unit::THREAD_X,
                                Grid::Unit::THREAD_Y})
                     .assumes(Grid::Dim::GRID_DIM_X == v),
                 error::UserError);

    // Try to constrain a dimension that does belong to this
    // function.
    ASSERT_NO_THROW(Tileable(v = Expr(0), TestDSCPU["dummy"], v,
                             Computes(
                                 Produces::Subset(TestDSCPU, {v, v}),
                                 Consumes(SubsetObjMany(s))))
                        .occupies({Grid::Unit::THREAD_X})
                        .assumes(Grid::Dim::BLOCK_DIM_X == v));

    // Ok now too.
    ASSERT_NO_THROW(Tileable(v = Expr(0), TestDSCPU["dummy"], v,
                             Computes(
                                 Produces::Subset(TestDSCPU, {v, v}),
                                 Consumes(SubsetObjMany(s))))
                        .occupies({Grid::Unit::THREAD_X,
                                   Grid::Unit::THREAD_Y})
                        .assumes(Grid::Dim::BLOCK_DIM_Y == v));

    // Constrain a child, ok too
    // Ok now too.
    ASSERT_NO_THROW(Tileable(v = Expr(0), TestDSCPU["dummy"], v,
                             Computes(
                                 Produces::Subset(TestDSCPU, {v, v}),
                                 Consumes(SubsetObjMany(s))))
                        .occupies({Grid::Unit::BLOCK_X,
                                   Grid::Unit::THREAD_Y})
                        .assumes(Grid::Dim::BLOCK_DIM_Y == v));
}