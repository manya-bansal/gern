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

    std::vector<Variable> getFields() const override {
        return {x, len};
    }
    Function getAllocateFunction() const override {
        return Function{
            .name = "allocate",
            .args = {x, len},
        };
    }
    Function getFreeFunction() const override {
        return Function{
            .name = "destroy",
            .args = {},
        };
    }
    Function getInsertFunction() const override {
        return Function{
            .name = "insert",
            .args = {x, len},
        };
    }
    Function getQueryFunction() const override {
        return Function{
            .name = "query",
            .args = {x, len},
        };
    }

private:
    std::string name;
    Variable x{"x"};
    Variable len{"len"};
};

class add : public AbstractFunction {
public:
    add()
        : input(new const ArrayCPU("input")),
          output(new const ArrayCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Pattern getAnnotation() override {
        Variable x("x");

        return For(x = Expr(0), end, step,
                   Produces::Subset(output, {x, step}),
                   Consumes::Subset(input, {x, step}));
    }

    virtual Function getFunction() override {
        Function f;
        f.name = "gern::impl::add";
        f.args = {Argument(input), Argument(output)};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-array.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
    Variable step{"step"};
};

class addTemplate : public add {
public:
    addTemplate()
        : add() {
    }
    std::string getName() {
        return "gern::impl::addTemplate";
    }

    virtual Function getFunction() override {
        Function f;
        f.name = "gern::impl::addTemplate";
        f.args = {Argument(input), Argument(output)};
        f.template_args = {step};
        return f;
    }
};

// This is perhaps a contrived example, but it exists to
// exercise the ability to add for loops inside
// the compute annotation.
class reduction : public AbstractFunction {
public:
    reduction()
        : input(new const annot::ArrayCPU("input")),
          output(new const annot::ArrayCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Pattern getAnnotation() override {
        Variable x("x");
        Variable r("r");
        Variable step("step");
        Variable end("end");

        return For(x = Expr(0), end, step,
                   Computes(
                       Produces::Subset(output, {x, step}),
                       Consumes::Subsets(
                           For(r = Expr(0), end, 1,
                               SubsetObjMany{
                                   SubsetObj(input, {r, 1})}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-array.h",
        };
    }

    virtual Function getFunction() override {
        Function f;
        f.name = "gern::impl::add";
        f.args = {Argument(input), Argument(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
};

}  // namespace annot
}  // namespace gern