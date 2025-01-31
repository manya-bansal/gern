#pragma once

#include <iostream>
#include <set>

namespace gern {

namespace Grid {
enum Unit {
    UNDEFINED,  // For variables that are not bound to the grid.

    SCALAR_UNIT,

    THREAD_ID_X,
    THREAD_ID_Y,
    THREAD_ID_Z,

    BLOCK_ID_X,
    BLOCK_ID_Y,
    BLOCK_ID_Z,
};

enum Dim {
    GRID_DIM_X,
    GRID_DIM_Y,
    GRID_DIM_Z,

    BLOCK_DIM_X,
    BLOCK_DIM_Y,
    BLOCK_DIM_Z,
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

}  // namespace gern