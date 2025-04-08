#pragma once

#include <iostream>
#include <set>

namespace gern {

namespace Grid {
enum Unit {
    UNDEFINED,  // For variables that are not bound to the grid.

    SCALAR_UNIT,

    THREAD_X_IN_WRAPS,
    THREAD_Y_IN_WRAPS,
    THREAD_Z_IN_WRAPS,

    THREAD_X,
    THREAD_Y,
    THREAD_Z,

    WARP,

    BLOCK_X,
    BLOCK_Y,
    BLOCK_Z,
};

enum Dim {
    GRID_DIM_X,
    GRID_DIM_Y,
    GRID_DIM_Z,

    BLOCK_DIM_X,
    BLOCK_DIM_Y,
    BLOCK_DIM_Z,

    WARP_DIM,
    WARP_DIM_X,
    WARP_DIM_Y,
    WARP_DIM_Z,

    NULL_DIM,
};

enum Level {
    NULL_LEVEL,

    SCALAR,
    THREADS,
    WARPS,
    BLOCK,
    BLOCK_CLUSTER,
    GRID,
};

}  // namespace Grid

std::ostream &operator<<(std::ostream &, const Grid::Unit &);
std::ostream &operator<<(std::ostream &, const Grid::Dim &);
std::ostream &operator<<(std::ostream &, const Grid::Level &);

bool isLegalUnit(const Grid::Unit &);
// Indicates whether the property chances over the
// same kernel launch.

/**
 * @brief isLegalLevel checks whether the grid unit is legal for the GPU.
 *
 * @param u unit to check.
 * @return true
 * @return false
 */
bool isLegalLevel(const Grid::Level &u);

/**
 * @brief This function checks whether the unit can be distributed over
 *        a grid's property.
 *
 * @param u
 * @param p
 * @return true
 * @return false
 */
bool legalToDistribute(const std::set<Grid::Unit> &u, const Grid::Unit &p);

/**
 * @brief Get the unit assosciated with a particular property.
 *
 * @param p
 * @return Grid::Level
 */
Grid::Level getLevel(const Grid::Unit &p);
Grid::Level getLevel(const std::set<Grid::Unit> &p);

Grid::Level getLevel(const Grid::Dim &dim);
Grid::Level getLevel(const std::set<Grid::Dim> &dims);

Grid::Dim getDim(const Grid::Unit &unit);
std::set<Grid::Dim> getDims(const std::set<Grid::Unit> &unit);

bool isDimInScope(const Grid::Dim &dim, const std::set<Grid::Dim> &dims);

}  // namespace gern