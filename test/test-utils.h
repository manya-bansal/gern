#ifndef GERN_TEST_UTILS_H
#define GERN_TEST_UTILS_H

#include "annotations/data_dependency_language.h"
#include <algorithm>

namespace gern {

template<typename T>
static std::string getStrippedString(T e) {
    std::stringstream ss;
    ss << e;
    auto str = ss.str();
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return std::string(str);
}

template<typename T>
static bool areDisjoint(std::set<T> s1, std::set<T> s2) {
    for (const auto &e : s1) {
        if (s2.find(e) != s2.end()) {
            return false;
        }
    }
    return true;
}

}  // namespace gern

#endif