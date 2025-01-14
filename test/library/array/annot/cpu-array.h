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
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::ArrayCPU::allocate",
            .args = {x, len},
        };
    }
    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "destroy",
            .args = {},
        };
    }
    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert",
            .args = {x, len},
        };
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, len},
        };
    }

protected:
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

        return For(x = Expr(0), output["size"], step,
                   Produces::Subset(output, {x, step}),
                   Consumes::Subset(input, {x, step}));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Parameter(input), Parameter(output)};
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

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::addTemplate";
        f.args = {Parameter(input), Parameter(output)};
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

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
};

}  // namespace annot
}  // namespace gern