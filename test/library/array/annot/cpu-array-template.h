#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace annot {

template<int Size>
class ArrayCPUTemplate : public AbstractDataType {
public:
    ArrayCPUTemplate(const std::string &name, bool temp = false)
        : name(name), temp(temp) {
    }
    ArrayCPUTemplate()
        : ArrayCPUTemplate("test") {
    }

    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        if (temp) {
            return "auto";  // Don't know what type it will be.
        }
        return "gern::impl::ArrayCPUTemplate<" + std::to_string(Size) + ">";
    }

    std::vector<Variable> getFields() const override {
        return {x, len};
    }
    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::temp_allocate",
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

    FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert",
            .args = {x},
            .template_args = {len},
        };
    }

    FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "destroy",
            .args = {},
        };
    }

    bool insertQuery() const override {
        return true;
    }

    bool freeAlloc() const override {
        return false;
    }

protected:
    std::string name;
    bool temp;
    Variable x{"x"};
    Variable len{"len"};
};

class addStaticStore : public AbstractFunction {
public:
    addStaticStore()
        : input(new const ArrayCPUTemplate<10>("input")),
          output(new const ArrayCPUTemplate<10>("output")) {
    }

    Annotation getAnnotation() override {
        Variable x("x");

        return annotate(Tileable(x = Expr(0), output["size"], step,
                                 Produces::Subset(output, {x, step}),
                                 Consumes::Subset(input, {x, step})));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Parameter(input), Parameter(output)};
        f.template_args = {step};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-array-template.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
    Variable step{"step"};
};

}  // namespace annot
}  // namespace gern