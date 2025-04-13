#pragma once

#include "compose/composable.h"
#include "compose/compose.h"
#include <string>

namespace gern {

using GernGenFuncPtr = void (*)(void **);

class Runner {
public:
    Runner(Composable c)
        : c(c) {
    }
    struct Options {
        std::string filename;
        std::string prefix = "/tmp";
        std::string cpp_std = "c++11";
        std::string include = "";
        std::string ldflags = "";
        std::string arch = "";
    };

    // Is there a better way to do default
    // init for a struct, while still
    // letting users set members by name?
    void compile(Options config);

    void evaluate(std::map<std::string, void *> args);

    FunctionSignature getSignature() const;

private:
    FunctionSignature signature;  // Function signature of the generated function.
    Composable c;
    GernGenFuncPtr fp;
    std::vector<std::string> argument_order;
    bool compiled = false;
};

}  // namespace gern