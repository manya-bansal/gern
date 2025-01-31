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
                             std::vector<T> v,
                             int ident,
                             std::string end = "") {
    int len = v.size();
    for (int i = 0; i < len; i++) {
        printIdent(os, ident);
        os << v[i];
        os << ((i != len - 1) ? "," : "");
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