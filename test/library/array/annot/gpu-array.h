#pragma once

#include "annotations/abstract_function.h"
#include "library/array/annot/cpu-array.h"

namespace gern {
namespace annot {

class ArrayGPU : public AbstractDataType {
public:
    ArrayGPU(const std::string &name)
        : name(name) {
    }
    ArrayGPU()
        : ArrayGPU("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "gern::impl::ArrayGPU";
    }

private:
    std::string name;
};

// This *must* be a device function.
class addGPU : public add {
public:
    addGPU()
        : add() {
    }

    std::vector<std::string> getHeader() {
        return {
            "gpu-array.h",
        };
    }

private:
    std::shared_ptr<ArrayGPU> input;
    std::shared_ptr<ArrayGPU> output;
    Variable end{"end"};
};

class reductionGPU : public reduction {
public:
    reductionGPU()
        : input(std::make_shared<ArrayGPU>("input")),
          output(std::make_shared<ArrayGPU>("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Variable r("r");
        Variable step("step");
        Variable end("end");

        return For(x = Expr(0), end, step,
                   Produces::Subset(output, {x, step}),
                   Consumes::Subsets(
                       For(r = Expr(0), end, 1,
                           {input, {r, 1}})));
    }

    std::vector<Argument> getArguments() {
        return {Argument(input), Argument(output)};
    }

    std::vector<std::string> getHeader() {
        return {
            "gpu-array.h",
        };
    }

    virtual Function getFunction() override {
        Function f;
        f.name = "gern::impl::add";
        f.args = {Argument(input), Argument(output)};
        return f;
    }

private:
    std::shared_ptr<ArrayGPU> input;
    std::shared_ptr<ArrayGPU> output;
};

}  // namespace annot
}  // namespace gern