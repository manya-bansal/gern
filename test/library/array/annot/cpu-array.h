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
    FunctionSignature newStageFunction() const {
        return FunctionSignature{
            .name = "new_stage",
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
    FunctionSignature getNewInsertFunction() const {
        return FunctionSignature{
            .name = "new_insert",
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

class add_1 : public AbstractFunction {
public:
    add_1()
        : input(new const ArrayCPU("input")),
          output(new const ArrayCPU("output")) {
    }

    Annotation getAnnotation() override {
        Variable x("x");

        return annotate(Tileable(x = Expr(0), output["size"], step,
                                 Produces::Subset(output, {x, step}),
                                 Consumes::Subset(input, {x, step})));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add_1";
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

class add_1_float : public add_1 {
public:
    add_1_float()
        : add_1() {
    }
    FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add_1_float";
        f.args = {Parameter(input), Parameter(output), Parameter(float_val)};
        return f;
    }

protected:
    Variable float_val{"float_val", Datatype::Float32};
};

class add1Template : public add_1 {
public:
    add1Template()
        : add_1() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add1Template";
        f.args = {Parameter(input), Parameter(output)};
        f.template_args = {step};
        return f;
    }
};

// Doesn't actually exist, there to
// exercise  test.
class addWithSize : public add_1 {
public:
    addWithSize()
        : add_1() {
    }
    std::string getName() {
        return "gern::impl::addWithSize";
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add1Template";
        f.args = {Parameter(input), Parameter(output), Parameter(step)};
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

    Annotation getAnnotation() override {
        Variable x("x");
        Variable r("r");
        Variable step("step");
        Variable reduce{"reduce"};
        // Variable reduce("reduce");

        return annotate(Tileable(x = Expr(0), output["size"], step,
                                 Computes(
                                     Produces::Subset(output, {x, step}),
                                     Consumes::Subsets(
                                         Reducible(r = Expr(0), k, reduce,
                                                   SubsetObjMany{
                                                       SubsetObj(input, {r, reduce})})))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-array.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::reduction";
        f.args = {Parameter(input), Parameter(output), Parameter(k)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable k{"k"};
};

}  // namespace annot
}  // namespace gern