#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace annot {

class MatrixCPU : public AbstractDataType {
public:
    MatrixCPU(const std::string &name)
        : name(name) {
    }
    MatrixCPU()
        : MatrixCPU("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::string getType() const override {
        return "gern::impl::MatrixCPU";
    }

    std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }

    Function getAllocateFunction() const override {
        return Function{
            .name = "allocate",
            .args = {x, y, l_x, l_y},
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
            .args = {x, y, l_x, l_y},
        };
    }
    Function getQueryFunction() const override {
        return Function{
            .name = "query",
            .args = {x, y, l_x, l_y},
        };
    }

private:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};
};

class MatrixAddCPU : public AbstractFunction {
public:
    MatrixAddCPU()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Pattern getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return For(x = Expr(0), row, l_x,
                   For(y = Expr(0), col, l_y,
                       Produces::Subset(input, {x, y, l_x, l_y}),
                       Consumes::Subset(output, {x, y, l_x, l_y})));
    }

    std::vector<Argument> getArguments() {
        return {
            Argument(input),
            Argument(output),
        };
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual Function getFunction() override {
        Function f;
        f.name = "gern::impl::add";
        f.args = {Argument(input), Argument(output)};
        return f;
    }

private:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

}  // namespace annot
}  // namespace gern