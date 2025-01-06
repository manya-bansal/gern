#ifndef GERN_ELEMENTWISE_FUNCTION_H
#define GERN_ELEMENTWISE_FUNCTION_H

#include "annotations/abstract_function.h"
#include "config.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace test {

class add : public AbstractFunction {
public:
    add()
        : input(std::make_shared<dummy::TestDSCPU>("input")),
          output(std::make_shared<dummy::TestDSCPU>("output")) {
    }
    std::string getName() {
        return "gern::lib::add";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Expr step(1);

        return For(x = Expr(0), end, step,
                   Computes(
                       Produces(
                           Subset(output, {x, step})),
                       Consumes(
                           Subset(input, {x, step}))));
    }

    std::vector<Argument> getArguments() {
        return {
            Argument(input),
            Argument(output),
        };
    }

    std::vector<std::string> getHeader() {
        return {
            "cpu-array-lib.h",
        };
    }

    std::vector<std::string> getIncludeFlags() {
        return {
            std::string(GERN_ROOT_DIR) + "/test/library/array/",
        };
    }

private:
    std::shared_ptr<dummy::TestDSCPU> input;
    std::shared_ptr<dummy::TestDSCPU> output;
    Variable end{"end"};
};

// This *must* be a device function.
class addGPU : public AbstractFunction {
public:
    addGPU()
        : input(std::make_shared<dummy::TestDSCPU>("input")),
          output(std::make_shared<dummy::TestDSCPU>("output")) {
    }
    std::string getName() {
        return "add";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Expr step(1);

        return For(x = Expr(0), end, step,
                   Computes(
                       Produces(
                           Subset(output, {x, step})),
                       Consumes(
                           Subset(input, {x, step}))));
    }

    std::vector<Argument> getArguments() {
        return {
            Argument(input),
            Argument(output),
        };
    }

    std::vector<std::string> getHeader() {
        return {
            "gpu-array-lib.h",
        };
    }

    std::vector<std::string> getIncludeFlags() {
        return {
            std::string(GERN_ROOT_DIR) + "/test/library/array/",
        };
    }

private:
    std::shared_ptr<dummy::TestDSCPU> input;
    std::shared_ptr<dummy::TestDSCPU> output;
    Variable end{"end"};
};

}  // namespace test
}  // namespace gern

#endif