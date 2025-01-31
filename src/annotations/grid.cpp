#include "annotations/grid.h"
#include "utils/error.h"

namespace gern {

// GCOVR_EXCL_START
std::ostream &operator<<(std::ostream &os, const Grid::Property &p) {
    switch (p) {

    case Grid::Property::BLOCK_ID_X:
        os << "BLOCK_ID_X";
        return os;
    case Grid::Property::BLOCK_ID_Y:
        os << "BLOCK_ID_Y";
        return os;
    case Grid::Property::BLOCK_ID_Z:
        os << "BLOCK_ID_Z";
        return os;

    case Grid::Property::THREAD_ID_X:
        os << "THREAD_ID_X";
        return os;
    case Grid::Property::THREAD_ID_Y:
        os << "THREAD_ID_Y";
        return os;
    case Grid::Property::THREAD_ID_Z:
        os << "THREAD_ID_Z";
        return os;

    case Grid::Property::BLOCK_DIM_X:
        os << "BLOCK_DIM_X";
        return os;
    case Grid::Property::BLOCK_DIM_Y:
        os << "BLOCK_DIM_Y";
        return os;
    case Grid::Property::BLOCK_DIM_Z:
        os << "BLOCK_DIM_Z";
        return os;

    default:
        os << "UNDEFINED";
        return os;
    }
}

std::ostream &operator<<(std::ostream &os, const Grid::Unit &unit) {
    switch (unit) {
    case Grid::Unit::GRID:
        os << "GRID";
        return os;
    case Grid::Unit::BLOCK_CLUSTER:
        os << "BLOCK_CLUSTER";
        return os;
    case Grid::Unit::BLOCK:
        os << "BLOCK";
        return os;
    case Grid::Unit::WARPS:
        os << "WARPS";
        return os;
    case Grid::Unit::THREADS:
        os << "THREADS";
        return os;
    default:
        os << "UNDEFINED";
        return os;
    }
}
// GCOVR_EXCL_STOP

bool isGridPropertySet(const Grid::Property &p) {
    return p != Grid::Property::UNDEFINED;
}

bool isLegalUnit(const Grid::Unit &u) {
    return u != Grid::Unit::NULL_UNIT;
}

bool isPropertyStable(const Grid::Property &p) {
    if (p == Grid::Property::BLOCK_DIM_X ||
        p == Grid::Property::BLOCK_DIM_Y ||
        p == Grid::Property::BLOCK_DIM_Z) {

        return true;
    }

    return false;
}

bool legalToDistribute(const Grid::Unit &u, const Grid::Property &p) {

    if (!isGridPropertySet(p) || !isLegalUnit(u)) {
        return false;
    }

    switch (u) {
    case Grid::Unit::BLOCK:
        if (p >= Grid::Property::BLOCK_ID_X) {
            return true;
        }
        break;
    case Grid::Unit::WARPS:
        if (p >= Grid::Property::BLOCK_ID_X) {  // Need to add a property for warp.
            return true;
        }
        break;
    case Grid::Unit::THREADS:
        if (p >= Grid::Property::THREAD_ID_X) {
            return true;
        }
        break;
    default:
        throw error::InternalError("This unit has not been implemented!");
    }

    return false;
}

Grid::Unit getUnit(const Grid::Property &p) {

    switch (p) {
    case Grid::Property::THREAD_ID_X:
    case Grid::Property::THREAD_ID_Y:
    case Grid::Property::THREAD_ID_Z:
        return Grid::Unit::THREADS;
    case Grid::Property::BLOCK_ID_X:
    case Grid::Property::BLOCK_ID_Y:
    case Grid::Property::BLOCK_ID_Z:
        return Grid::Unit::BLOCK;
    default:
        return Grid::Unit::NULL_UNIT;
    }
}

}  // namespace gern