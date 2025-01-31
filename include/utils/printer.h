#pragma once

#include <iostream>

namespace gern {
namespace util {

static void printIdent(std::ostream &os, int ident) {
    for (int i = 0; i < ident; i++) {
        os << "  ";
    }
}

template<typename T>
static void iterable_printer(std::ostream &os,
                             const T &v,
                             int ident,
                             std::string end = "") {
    for (const auto &e : v) {
        printIdent(os, ident);
        os << e;
        // os << ((i != len - 1) ? "," : "");
        os << end;
    }
}

template<typename T>
std::string str(T t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

}  // namespace util
}  // namespace gern