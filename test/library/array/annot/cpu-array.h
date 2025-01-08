#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace annot {

class ArrayCPU : public AbstractDataType {
public:
    ArrayCPU(const std::string &name)
        : name(name) {
    }
    ArrayCPU()
        : ArrayCPU("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "gern::impl::ArrayCPU";
    }

private:
    std::string name;
};

class add : public AbstractFunction {
public:
    add()
        : input(std::make_shared<ArrayCPU>("input")),
          output(std::make_shared<ArrayCPU>("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Pattern getAnnotation() {
        Variable x("x");
        Variable step("step");

        return For(x = Expr(0), end, step,
                   Computes(
                       Produces::Subset(output, {x, step}),
                       Consumes(
                           SubsetObj(input, {x, step}))));
    }

    std::vector<Argument> getArguments() {
        return {
            Argument(input),
            Argument(output),
        };
    }

    std::vector<std::string> getHeader() {
        return {
            "cpu-array.h",
        };
    }

private:
    std::shared_ptr<ArrayCPU> input;
    std::shared_ptr<ArrayCPU> output;
    Variable end{"end"};
};

// This is perhaps a contrived example, but it exists to
// exercise the ability to add for loops inside
// the compute annotation.
class reduction : public AbstractFunction {
public:
    reduction()
        : input(std::make_shared<annot::ArrayCPU>("input")),
          output(std::make_shared<annot::ArrayCPU>("output")) {
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
                   Computes(
                       Produces::Subset(output, {x, step}),
                       Consumes(
                           For(r = Expr(0), end, 1,
                               Subsets{
                                   SubsetObj(input, {r, 1})}))));
    }

    std::vector<Argument> getArguments() {
        return {Argument(input), Argument(output)};
    }

    std::vector<std::string> getHeader() {
        return {
            "cpu-array.h",
        };
    }

private:
    std::shared_ptr<annot::ArrayCPU> input;
    std::shared_ptr<annot::ArrayCPU> output;
};

}  // namespace annot
}  // namespace gern