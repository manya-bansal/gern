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

    case Grid::Unit::THREAD_X:
        os << "THREAD_X";
        return os;
    case Grid::Unit::THREAD_Y:
        os << "THREAD_Y";
        return os;
    case Grid::Unit::THREAD_Z:
        os << "THREAD_Z";
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

Grid::Level getLevel(const Grid::Unit &unit) {

    switch (unit) {
    case Grid::Unit::THREAD_X:
    case Grid::Unit::THREAD_Y:
    case Grid::Unit::THREAD_Z:
        return Grid::Level::THREADS;
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
        throw error::UserError("Unimplemented Dimension");
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

Grid::Dim getDim(const Grid::Unit &unit) {

    switch (unit) {
    case Grid::Unit::THREAD_X:
        return Grid::Dim::BLOCK_DIM_X;
    case Grid::Unit::THREAD_Y:
        return Grid::Dim::BLOCK_DIM_Y;
    case Grid::Unit::THREAD_Z:
        return Grid::Dim::BLOCK_DIM_Z;
    case Grid::Unit::BLOCK_X:
        return Grid::Dim::GRID_DIM_X;
    case Grid::Unit::BLOCK_Y:
        return Grid::Dim::GRID_DIM_Y;
    case Grid::Unit::BLOCK_Z:
        return Grid::Dim::GRID_DIM_Z;
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

}  // namespace gern