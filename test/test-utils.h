#pragma once

#include "annotations/data_dependency_language.h"
#include "compose/runner.h"
#include "config.h"

namespace gern {
namespace test {

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

[[maybe_unused]] static gern::Runner::Options cpuRunner(std::string dir) {
    gern::Runner::Options o;
    o.filename = "test";
    o.prefix = "/tmp";
    o.include = " -I " + std::string(GERN_ROOT_DIR) +
                "/test/library/" + dir + "/impl";
    return o;
}

}  // namespace test
}  // namespace gern