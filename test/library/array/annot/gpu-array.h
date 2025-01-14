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

    std::vector<Variable> getFields() const override {
        return {x, len};
    }
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::ArrayGPU::allocate",
            .args = {x, len},
        };
    }
    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "destroy",
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
};

class reductionGPU : public reduction {
public:
    reductionGPU()
        : input(new const ArrayGPU("input")),
          output(new const ArrayGPU("output")) {
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
                   Produces::Subset(output, {x, step}),
                   Consumes::Subsets(
                       For(r = Expr(0), end, 1,
                           {input, {r, 1}})));
    }

    std::vector<std::string> getHeader() override {
        return {
            "gpu-array.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Argument(input), Argument(output)};
        return f;
    }

private:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
};

class ArrayStaticGPU : public ArrayGPU {
public:
    ArrayStaticGPU(const std::string &name)
        : ArrayGPU(name) {
    }

    std::vector<Variable> getFields() const override {
        return {x, len};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::allocate_local",
            .template_args = {len},
        };
    }
    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "dummy",
        };
    }

    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert_array",
            .args = {x},
            .template_args = {len},
        };
    }

    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x},
            .template_args = {len},
        };
    }

    bool freeAlloc() const override {
        return false;
    }

    bool insertQuery() const override {
        return true;
    }
};

class addGPUTemplate : public addGPU {
public:
    addGPUTemplate()
        : addGPU() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Argument(input), Argument(output)};
        f.template_args = {step};
        return f;
    }
};

}  // namespace annot
}  // namespace gern