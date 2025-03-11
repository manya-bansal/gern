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
        Variable i("i");
        Variable j("j");
        Variable k("k");

        Variable ti("ti", Datatype::Int64);
        Variable tj("tj", Datatype::Int64);
        Variable tk("tk", Datatype::Int64);

        // return annotate(
        //     For(i = Expr(0), C["row"], ti,
        //         For(j = Expr(0), C["col"], tj,
        //             Produces::Subset(C, /* TODO */),
        //             Consumes::Subsets(
        //                 SubsetObjMany({
        //                     SubsetObj(A, /* TODO */),
        //                     SubsetObj(B, /* TODO */),
        //                 })))));

        return MatrixMultiplyCPU::getAnnotation();
    }

    FunctionSignature getFunction() override {
        return MatrixMultiplyCPU::getFunction();
    }
};
}  // namespace annot
}  // namespace gern
