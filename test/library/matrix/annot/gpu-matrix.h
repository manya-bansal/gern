#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "test-utils.h"
#include <iostream>

namespace gern {
namespace annot {

class MatrixGPU : public AbstractDataType {
public:
    MatrixGPU(const std::string &name)
        : name(name) {
    }
    MatrixGPU()
        : MatrixGPU("test") {
    }
    std::string getName() const override {
        return name;
    }

    std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }

    std::string getType() const override {
        return "gern::impl::MatrixGPU";
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

class MatrixAddGPU : public MatrixAddCPU {
public:
    MatrixAddGPU()
        : MatrixAddCPU() {
    }

    std::vector<std::string> getHeader() {
        return {
            "gpu-matrix.h",
        };
    }
};

}  // namespace annot
}  // namespace gern