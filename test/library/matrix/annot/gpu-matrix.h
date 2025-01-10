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

    std::string getType() const override {
        return "gern::impl::MatrixGPU";
    }

private:
    std::string name;
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