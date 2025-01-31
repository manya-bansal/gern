#include "annotations/grid.h"
#include "utils/error.h"
#include <set>

namespace gern {

// GCOVR_EXCL_START
std::ostream &operator<<(std::ostream &os, const Grid::Unit &p) {
    switch (p) {

    case Grid::Unit::BLOCK_ID_X:
        os << "BLOCK_ID_X";
        return os;
    case Grid::Unit::BLOCK_ID_Y:
        os << "BLOCK_ID_Y";
        return os;
    case Grid::Unit::BLOCK_ID_Z:
        os << "BLOCK_ID_Z";
        return os;

    case Grid::Unit::THREAD_ID_X:
        os << "THREAD_ID_X";
        return os;
    case Grid::Unit::THREAD_ID_Y:
        os << "THREAD_ID_Y";
        return os;
    case Grid::Unit::THREAD_ID_Z:
        os << "THREAD_ID_Z";
        return os;
    case Grid::Unit::SCALAR_UNIT:
        os << "SCALAR_UNIT";
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

    // Cannot distribute again.
    if (units.contains(distribute_unit)) {
        return false;
    }

    if (getLevel(units) > getLevel(distribute_unit)) {
        return false;
    }

    return true;
}

Grid::Level getLevel(const Grid::Unit &p) {

    switch (p) {
    case Grid::Unit::THREAD_ID_X:
    case Grid::Unit::THREAD_ID_Y:
    case Grid::Unit::THREAD_ID_Z:
        return Grid::Level::THREADS;
    case Grid::Unit::BLOCK_ID_X:
    case Grid::Unit::BLOCK_ID_Y:
    case Grid::Unit::BLOCK_ID_Z:
        return Grid::Level::BLOCK;
    case Grid::Unit::UNDEFINED:
        return Grid::Level::NULL_LEVEL;
    default:
        return Grid::Level::SCALAR;
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

}  // namespace gern