#pragma once

#include <iostream>

namespace gern {

class Grid {
public:
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

        THREAD_DIM_X,
        THREAD_DIM_Y,
        THREAD_DIM_Z,

        UNDEFINED,  // For variables that are not bound to the grid.
    };
};

std::ostream &operator<<(std::ostream &, const Grid::Property &);

bool isGridPropertySet(const Grid::Property &);
// Indicates whether the property chances over the
// same kernel launch.
bool isPropertyStable(const Grid::Property &);

}  // namespace gern