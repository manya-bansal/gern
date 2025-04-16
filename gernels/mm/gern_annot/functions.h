#pragma once

#include "adt.h"

namespace annot {

class MatrixMultiply : public AbstractFunction {
public:
    MatrixMultiply(AbstractDataTypePtr A,
                   AbstractDataTypePtr B,
                   AbstractDataTypePtr C)
        : A(A), B(B), C(C) {
        // k = k.bindToInt64(K);
    }
    Annotation getAnnotation() override {
        Variable i("i");
        Variable j("j");
        Variable k("k");

        Variable ti("ti", Datatype::Int64, true);
        Variable tj("tj", Datatype::Int64, true);
        Variable tk("tk", Datatype::Int64, true);

        return Tileable(i = Expr(0), ADTMember(C, "row", true), ti,
                        Tileable(j = Expr(0), ADTMember(C, "col", true), tj,
                                 Produces::Subset(C, {i, j, ti, tj}),
                                 Consumes::Subsets(
                                     Reducible(k = Expr(0), k_dim, tk,
                                               SubsetObjMany({
                                                   SubsetObj(A, {i, k, ti, tk}),
                                                   SubsetObj(B, {k, j, tk, tj}),
                                               })

                                                   ))))
            .occupies({Grid::Unit::SCALAR_UNIT});
    }
    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "matrix_multiply";
        f.args = {
            Parameter(A),
            Parameter(B),
            Parameter(C),
        };
        f.template_args = {k_dim};
        return f;
    }
    std::vector<std::string> getHeader() override {
        return {
            "impl/matrix-gpu.h",
            "impl/matrix_multiply.h",
        };
    }

protected:
    AbstractDataTypePtr A;
    AbstractDataTypePtr B;
    AbstractDataTypePtr C;
    Variable k_dim{"k_dim", Datatype::Int64, true};
};

class MatrixMultiplyWarp : public MatrixMultiply {
public:
    MatrixMultiplyWarp(AbstractDataTypePtr A,
                       AbstractDataTypePtr B,
                       AbstractDataTypePtr C)
        : MatrixMultiply(A, B, C) {
    }
    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "matrix_multiply_warp";
        f.args = {
            Parameter(A),
            Parameter(B),
            Parameter(C),
        };
        f.template_args = {k_dim};
        return f;
    }
};

class MatrixMultiplySync : public MatrixMultiply {
public:
    MatrixMultiplySync(AbstractDataTypePtr A,
                       AbstractDataTypePtr B,
                       AbstractDataTypePtr C)
        : MatrixMultiply(A, B, C) {
    }
    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "matrix_multiply_sync";
        f.args = {
            Parameter(A),
            Parameter(B),
            Parameter(C),
        };
        f.template_args = {k_dim};
        return f;
    }
    std::vector<std::string> getHeader() override {
        return {
            "impl/matrix-gpu.h",
            "impl/matrix_multiply.h",
            "impl/column-major.h",
        };
    }
};

class MatrixMultiplyReg : public MatrixMultiply {
public:
    MatrixMultiplyReg(AbstractDataTypePtr A,
                      AbstractDataTypePtr B,
                      AbstractDataTypePtr C)
        : MatrixMultiply(A, B, C) {
    }
    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "matrix_multiply_reg";
        f.args = {
            Parameter(A),
            Parameter(B),
            Parameter(C),
        };
        f.template_args = {k_dim};
        return f;
    }
};

}  // namespace annot
