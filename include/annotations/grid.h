#pragma once

#include <iostream>

namespace gern {

namespace Grid {
enum Property {
    UNDEFINED,  // For variables that are not bound to the grid.

    THREAD_ID_X,
    THREAD_ID_Y,
    THREAD_ID_Z,

    BLOCK_DIM_X,
    BLOCK_DIM_Y,
    BLOCK_DIM_Z,

    BLOCK_ID_X,
    BLOCK_ID_Y,
    BLOCK_ID_Z,
};

enum Unit {
    NULL_UNIT,

    SCALAR,
    THREADS,
    WARPS,
    BLOCK,
    BLOCK_CLUSTER,
    GRID,
};

}  // namespace Grid

std::ostream &operator<<(std::ostream &, const Grid::Property &);
std::ostream &operator<<(std::ostream &, const Grid::Unit &);

bool isGridPropertySet(const Grid::Property &);
// Indicates whether the property chances over the
// same kernel launch.
bool isPropertyStable(const Grid::Property &);

/**
 * @brief isLegalUnit checks whether the grid unit is legal for the GPU.
 *
 * @param u unit to check.
 * @return true
 * @return false
 */
bool isLegalUnit(const Grid::Unit &u);

/**
 * @brief This function checks whether the unit can be distributed over
 *        a grid's property.
 *
 * @param u
 * @param p
 * @return true
 * @return false
 */
bool legalToDistribute(const Grid::Unit &u, const Grid::Property &p);

/**
 * @brief Get the unit assosciated with a particular property.
 *
 * @param p
 * @return Grid::Unit
 */
Grid::Unit getUnit(const Grid::Property &p);

}  // namespace gern