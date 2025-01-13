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
    ArrayCPUTemplate(const std::string &name)
        : name(name) {
    }
    ArrayCPUTemplate()
        : ArrayCPUTemplate("test") {
    }

    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "gern::impl::ArrayCPUTemplate<" + std::to_string(Size) + ">";
    }

    std::vector<Variable> getFields() const override {
        return {x, len};
    }
    Function getAllocateFunction() const override {
        return Function{
            .name = "gern::impl::temp_allocate",
            .template_args = {len},
        };
    }

    Function getQueryFunction() const override {
        return Function{
            .name = "query",
            .args = {x},
            .template_args = {len},
        };
    }

    Function getInsertFunction() const override {
        return Function{
            .name = "insert",
            .args = {x},
            .template_args = {len},
        };
    }

    Function getFreeFunction() const override {
        return Function{
            .name = "destroy",
            .args = {},
        };
    }

    bool insertQuery() const override {
        return true;
    }

protected:
    std::string name;
    Variable x{"x"};
    Variable len{"len"};
};

class addStaticStore : public AbstractFunction {
public:
    addStaticStore()
        : input(new const ArrayCPUTemplate<10>("input")),
          output(new const ArrayCPUTemplate<10>("output")) {
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