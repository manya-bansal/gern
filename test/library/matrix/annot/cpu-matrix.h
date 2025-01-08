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

private:
    std::string name;
};

class MatrixAddCPU : public AbstractFunction {
public:
    MatrixAddCPU()
        : input(std::make_shared<MatrixCPU>("input")),
          output(std::make_shared<MatrixCPU>("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Pattern getAnnotation() {

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

    std::vector<std::string> getHeader() {
        return {
            "cpu-matrix.h",
        };
    }

private:
    std::shared_ptr<MatrixCPU> input;
    std::shared_ptr<MatrixCPU> output;
    Variable end{"end"};
};

}  // namespace annot
}  // namespace gern