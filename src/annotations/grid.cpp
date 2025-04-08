#include "annotations/grid.h"
#include "utils/error.h"
#include <set>

namespace gern {

// GCOVR_EXCL_START
std::ostream &operator<<(std::ostream &os, const Grid::Unit &p) {
    switch (p) {

    case Grid::Unit::BLOCK_X:
        os << "BLOCK_X";
        return os;
    case Grid::Unit::BLOCK_Y:
        os << "BLOCK_Y";
        return os;
    case Grid::Unit::BLOCK_Z:
        os << "BLOCK_Z";
        return os;
    case Grid::Unit::WARP_X:
        os << "WARP_X";
        return os;
    case Grid::Unit::WARP_Y:
        os << "WARP_Y";
        return os;
    case Grid::Unit::WARP_Z:
        os << "WARP_X";
        return os;
    case Grid::Unit::THREAD_X:
        os << "THREAD_X";
        return os;
    case Grid::Unit::THREAD_Y:
        os << "THREAD_Y";
        return os;
    case Grid::Unit::THREAD_Z:
        os << "THREAD_Z";
        return os;
    case Grid::Unit::THREAD_X_IN_WRAPS:
        os << "THREAD_X_IN_WRAPS";
        return os;
    case Grid::Unit::THREAD_Y_IN_WRAPS:
        os << "THREAD_Y_IN_WRAPS";
        return os;
    case Grid::Unit::THREAD_Z_IN_WRAPS:
        os << "THREAD_Z_IN_WRAPS";
        return os;
    case Grid::Unit::SCALAR_UNIT:
        os << "SCALAR_UNIT";
        return os;
    default:
        os << "UNDEFINED";
        return os;
    }
}

std::ostream &operator<<(std::ostream &os, const Grid::Dim &dim) {
    switch (dim) {

    case Grid::Dim::BLOCK_DIM_X:
        os << "BLOCK_DIM_X";
        return os;
    case Grid::Dim::BLOCK_DIM_Y:
        os << "BLOCK_DIM_Y";
        return os;
    case Grid::Dim::BLOCK_DIM_Z:
        os << "BLOCK_DIM_Z";
        return os;
    case Grid::Dim::GRID_DIM_X:
        os << "GRID_DIM_X";
        return os;
    case Grid::Dim::GRID_DIM_Y:
        os << "GRID_DIM_Y";
        return os;
    case Grid::Dim::GRID_DIM_Z:
        os << "GRID_DIM_Z";
        return os;
    case Grid::Dim::NULL_DIM:
        os << "NULL_DIM";
        return os;
    default:
        error::InternalError("Unreachable");
        return os;
    }
}

std::ostream &operator<<(std::ostream &os, const Grid::Level &unit) {
    switch (unit) {
    case Grid::Level::GRID:
        os << "GRID";
        return os;
    case Grid::Level::BLOCK_CLUSTER:
        os << "BLOCK_CLUSTER";
        return os;
    case Grid::Level::BLOCK:
        os << "BLOCK";
        return os;
    case Grid::Level::WARPS:
        os << "WARPS";
        return os;
    case Grid::Level::THREADS:
        os << "THREADS";
        return os;
    case Grid::Level::SCALAR:
        os << "SCALAR";
        return os;
    default:
        os << "UNDEFINED";
        return os;
    }
}
// GCOVR_EXCL_STOP

bool isLegalUnit(const Grid::Unit &p) {
    return p > Grid::Unit::SCALAR_UNIT;
}

bool isLegalLevel(const Grid::Level &u) {
    return u != Grid::Level::NULL_LEVEL;
}

bool legalToDistribute(const std::set<Grid::Unit> &units, const Grid::Unit &distribute_unit) {
    // If it is undefined, then complain.
    if (!isLegalUnit(distribute_unit)) {
        return false;
    }

    if (getLevel(units) > getLevel(distribute_unit)) {
        return false;
    }

    return true;
}

Grid::Level getLevel(const Grid::Unit &unit) {

    switch (unit) {
    case Grid::Unit::THREAD_X:
    case Grid::Unit::THREAD_Y:
    case Grid::Unit::THREAD_Z:
    case Grid::Unit::THREAD_X_IN_WRAPS:
    case Grid::Unit::THREAD_Y_IN_WRAPS:
    case Grid::Unit::THREAD_Z_IN_WRAPS:
        return Grid::Level::THREADS;
    case Grid::Unit::WARP_X:
    case Grid::Unit::WARP_Y:
    case Grid::Unit::WARP_Z:
        return Grid::Level::WARPS;
    case Grid::Unit::BLOCK_X:
    case Grid::Unit::BLOCK_Y:
    case Grid::Unit::BLOCK_Z:
        return Grid::Level::BLOCK;
    case Grid::Unit::UNDEFINED:
        return Grid::Level::NULL_LEVEL;
    default:
        return Grid::Level::SCALAR;
    }
}

Grid::Level getLevel(const Grid::Dim &dim) {

    switch (dim) {
    case Grid::Dim::BLOCK_DIM_X:
    case Grid::Dim::BLOCK_DIM_Y:
    case Grid::Dim::BLOCK_DIM_Z:
        return Grid::Level::THREADS;
    case Grid::Dim::GRID_DIM_X:
    case Grid::Dim::GRID_DIM_Y:
    case Grid::Dim::GRID_DIM_Z:
        return Grid::Level::BLOCK;
    default:
        return Grid::Level::NULL_LEVEL;
    }
}

Grid::Level getLevel(const std::set<Grid::Unit> &units) {
    Grid::Level level{Grid::Level::NULL_LEVEL};
    for (const auto &unit : units) {
        auto current_unit = getLevel(unit);
        level = (level > current_unit) ? level : current_unit;
    }
    return level;
}

Grid::Level getLevel(const std::set<Grid::Dim> &dims) {
    Grid::Level level{Grid::Level::NULL_LEVEL};
    for (const auto &dim : dims) {
        auto current_level = getLevel(dim);
        level = (level > current_level) ? level : current_level;
    }
    return level;
}

Grid::Dim getDim(const Grid::Unit &unit) {

    switch (unit) {
    case Grid::Unit::THREAD_X:
        return Grid::Dim::BLOCK_DIM_X;
    case Grid::Unit::THREAD_Y:
        return Grid::Dim::BLOCK_DIM_Y;
    case Grid::Unit::THREAD_Z:
        return Grid::Dim::BLOCK_DIM_Z;
    case Grid::Unit::THREAD_X_IN_WRAPS:
        return Grid::Dim::WARP_DIM_X;
    case Grid::Unit::THREAD_Y_IN_WRAPS:
        return Grid::Dim::WARP_DIM_Y;
    case Grid::Unit::THREAD_Z_IN_WRAPS:
        return Grid::Dim::WARP_DIM_Z;
    case Grid::Unit::WARP_X:
        return Grid::Dim::WARP_DIM_X;
    case Grid::Unit::WARP_Y:
        return Grid::Dim::WARP_DIM_Y;
    case Grid::Unit::WARP_Z:
        return Grid::Dim::WARP_DIM_Z;
    case Grid::Unit::BLOCK_X:
        return Grid::Dim::GRID_DIM_X;
    case Grid::Unit::BLOCK_Y:
        return Grid::Dim::GRID_DIM_Y;
    case Grid::Unit::BLOCK_Z:
        return Grid::Dim::GRID_DIM_Z;
    case Grid::Unit::UNDEFINED:
    case Grid::Unit::SCALAR_UNIT:
        return Grid::Dim::NULL_DIM;
    default:
        throw error::UserError("Asking for the dim for an undefined unit!");
    }
}

std::set<Grid::Dim> getDims(const std::set<Grid::Unit> &units) {
    std::set<Grid::Dim> dims;
    for (const auto &unit : units) {
        dims.insert(getDim(unit));
    }
    return dims;
}

bool isDimInScope(const Grid::Dim &dim, const std::set<Grid::Dim> &dims) {

    // If this is a dimension that the units directly control, then it is in scope.
    if (dims.contains(dim)) {
        return true;
    }

    // If not under direct control, check if the dim belongs to a unit at a lower level.
    if (getLevel(dim) < getLevel(dims)) {
        return true;
    }

    return false;  // Cannot control.
}

}  // namespace gern