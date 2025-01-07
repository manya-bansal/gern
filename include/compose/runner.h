#pragma once

#include "compose/pipeline.h"

namespace gern {

using GernGenFuncPtr = void (*)(void **);

class Runner {
public:
    Runner(const Pipeline &p)
        : p(p) {
    }
    struct Options {
        std::string filename = "gern_file";
        std::string prefix = "/tmp";
        std::string include = "";
        std::string ldflags = "";
        std::string arch = "";
    };

    // Is there a better way to do default
    // init for a struct, while still
    // letting users set members by name?
    void compile(Options config);

    void evaluate(std::map<std::string, void *> args);

private:
    Pipeline p;
    GernGenFuncPtr fp;
    std::vector<std::string> argument_order;
    bool compiled = false;
};

}  // namespace gern