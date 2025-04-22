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
class Add1GPU : public add_1 {
public:
    Add1GPU()
        : add_1() {
    }

    Annotation getAnnotation() override {
        Variable x("x");

        return (Tileable(x = Expr(0), output["size"], step,
                         Produces::Subset(output, {x, step}),
                         Consumes::Subset(input, {x, step}))
                    .occupies({Grid::Unit::SCALAR_UNIT}));
    }

    std::vector<std::string> getHeader() override {
        return {
            "gpu-array.h",
            "gpu-array.cu",
        };
    }
};

class reductionGPU : public reduction {
public:
    reductionGPU()
        : input(new const ArrayGPU("input")),
          output(new const ArrayGPU("output")) {
    }

    Annotation getAnnotation() override {
        Variable x("x");
        Variable r("r");
        Variable step("step");
        Variable end("end");
        Variable extra("extra");

        return (Tileable(x = Expr(0), output["size"], step,
                         Produces::Subset(output, {x, step}),
                         Consumes::Subsets(
                             Reducible(r = Expr(0), k, end,
                                       {input, {r, k}}))))
            .occupies({Grid::Unit::SCALAR_UNIT});
    }

    std::vector<std::string> getHeader() override {
        return {
            "gpu-array.h",
            "gpu-array.cu",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::reduction";
        f.args = {Parameter(input), Parameter(output), Parameter(k)};
        return f;
    }

private:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable k{"k"};
};

class ArrayStaticGPU : public ArrayGPU {
public:
    ArrayStaticGPU(const std::string &name, bool temp = false)
        : name(name), temp(temp) {
    }

    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        if (temp) {
            return "auto";
        }
        return "gern::impl::ArrayGPU";
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

private:
    std::string name;
    bool temp;
    Variable x{"x"};
    Variable len{"len"};
};

class Add1GPUTemplate : public Add1GPU {
public:
    Add1GPUTemplate()
        : Add1GPU() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add_1";
        f.args = {Parameter(input), Parameter(output)};
        f.template_args = {step};
        return f;
    }
};

class AddArrayThreads : public Add1GPU {
public:
    AddArrayThreads() = default;
    Annotation getAnnotation() override {
        return resetUnit(Add1GPU::getAnnotation(),
                         {Grid::Unit::THREAD_X})
            .assumes(GridDim(Grid::Dim::BLOCK_DIM_X) >= step);
    }
};

}  // namespace annot
}  // namespace gern