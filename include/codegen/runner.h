#pragma once

#include "compose/pipeline.h"

namespace gern {

class Runner {
public:
    Runner(const Pipeline &p)
        : p(p) {
    }
    struct Options {
        std::string compiler;
        std::string filename;
        std::string prefix;
        std::string includes;
        std::string ldflags;
    };

    void compile(Options config = Options{"g++", "test.cpp", "/tmp", "", ""});
    void evaluate(std::map<std::string, void *> args);

private:
    Pipeline p;
    GernGenFuncPtr fp;
    std::vector<std::string> argument_order;
    bool compiled = false;
    bool lowered = false;
};

}  // namespace gern