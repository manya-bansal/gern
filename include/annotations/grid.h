#pragma once

#include <iostream>

namespace gern {

namespace Grid {
enum Property {

    BLOCK_ID_X,
    BLOCK_ID_Y,
    BLOCK_ID_Z,

    THREAD_ID_X,
    THREAD_ID_Y,
    THREAD_ID_Z,

    BLOCK_DIM_X,
    BLOCK_DIM_Y,
    BLOCK_DIM_Z,

    UNDEFINED,  // For variables that are not bound to the grid.
};

enum Unit {
    GRID,
    BLOCK_CLUSTER,
    BLOCK,
    WARPS,
    THREADS,
    NULL_UNIT,
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

}  // namespace gern