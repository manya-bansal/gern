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

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::MatrixGPU::allocate",
            .args = {x, y, l_x, l_y},
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
            .args = {x, y, l_x, l_y},
        };
    }
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, y, l_x, l_y},
        };
    }

protected:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"row"};
    Variable l_y{"col"};
};

class MatrixGPUStageSmem : public MatrixGPU {
public:
    MatrixGPUStageSmem(const std::string &name)
        : MatrixGPU(name) {
    }
    // Just override the getQueryFunction to return the stage_into_smem function.
    FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "stage_into_smem",
            .args = {x, y, l_x, l_y},
        };
    }

    bool insertQuery() const override {
        return true;
    }
};

class MatrixAddGPU : public MatrixAddCPU {
public:
    MatrixAddGPU()
        : MatrixAddCPU() {
    }

    Annotation getAnnotation() override {
        return resetUnit(MatrixAddCPU::getAnnotation(),
                         {Grid::Unit::SCALAR_UNIT});
    }

    std::vector<std::string> getHeader() {
        return {
            "gpu-matrix.h",
        };
    }
};

class MatrixAddGPUBlocks : public MatrixAddCPU {
public:
    MatrixAddGPUBlocks()
        : MatrixAddCPU() {
    }

    Annotation getAnnotation() override {
        return resetUnit(MatrixAddCPU::getAnnotation(),
                         {Grid::Unit::THREAD_X, Grid::Unit::THREAD_Y});
    }

    FunctionSignature getFunction() override {
        return FunctionSignature{
            .name = "gern::impl::add_smem",
            .args = {Parameter(input), Parameter(output)},
        };
    }

    std::vector<std::string> getHeader() override {
        return {
            "gpu-matrix.h",
        };
    }
};

}  // namespace annot
}  // namespace gern