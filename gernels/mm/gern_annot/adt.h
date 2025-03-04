#pragma once

#include <cstdlib>
#include <cstring>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <iostream>

#include "annotations/abstract_function.h"
#include "annotations/data_dependency_language.h"
#include "compose/composable.h"

using namespace gern;

namespace annot {

class MatrixGPU : public AbstractDataType {
public:
    MatrixGPU(int64_t rows, int64_t cols)
        : rows(rows), cols(cols) {
    }
};

}  // namespace annot
