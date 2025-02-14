#pragma once

#include "annotations/expr_nodes.h"

// Defining an std::less overload so that
// std::map<Variable, ....> in Stmt doesn't need to
// take in a special struct. The < operator in Expr
// is already overloaded, so it's not possible to use
// the usual std::less definition.
namespace std {
template<>
struct less<gern::Variable> {
    bool operator()(const gern::Variable &a, const gern::Variable &b) const {
        return a.getName() < b.getName();
    }
};

template<>
struct less<gern::ADTMember> {
    bool operator()(const gern::ADTMember &a, const gern::ADTMember &b) const {
        if (a.getDS() < b.getDS()) return true;   // Compare primary key
        if (b.getDS() < a.getDS()) return false;  // Compare reverse order
        return a.getMember() < b.getMember();     // Compare secondary key
    }
};

template<>
struct less<gern::Expr> {
    bool operator()(const gern::Expr &a, const gern::Expr &b) const {
        if (gern::isa<gern::ADTMember>(a) && gern::isa<gern::ADTMember>(b)) {
            return std::less<gern::ADTMember>()(gern::to<gern::ADTMember>(a), gern::to<gern::ADTMember>(b));
        }
        return a.ptr < b.ptr;
    }
};

template<>
struct less<gern::AbstractDataTypePtr> {
    bool operator()(const gern::AbstractDataTypePtr &a, const gern::AbstractDataTypePtr &b) const {
        return a.ptr < b.ptr;
    }
};

}  // namespace std