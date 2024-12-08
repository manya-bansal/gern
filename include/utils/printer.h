#ifndef GERN_PRINTER_UTILS_H
#define GERN_PRINTER_UTILS_H

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

#endif
