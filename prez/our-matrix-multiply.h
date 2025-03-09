#pragma once

#include "annotations/abstract_function.h"
#include "library/matrix/annot/cpu-matrix.h"
#include <iostream>

namespace gern {
namespace annot {

class OurMatrixMultiply : public MatrixMultiplyCPU {
public:
    OurMatrixMultiply()
        : MatrixMultiplyCPU() {
    }

    Annotation getAnnotation() override {
        return MatrixMultiplyCPU::getAnnotation();
    }

    FunctionSignature getFunction() override {
        return MatrixMultiplyCPU::getFunction();
    }
};
}  // namespace annot
}  // namespace gern
