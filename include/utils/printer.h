#pragma once

#include <iostream>

namespace gern {
namespace util {

static void printIdent(std::ostream &os, int ident) {
    for (int i = 0; i < ident; i++) {
        os << "  ";
    }
}

}  // namespace util
}  // namespace gern