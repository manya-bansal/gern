#pragma once

#include "annotations/abstract_function.h"
#include <iostream>

namespace gern {
namespace annot {

class MatrixBase : public AbstractDataType {
protected:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};
public:

    MatrixBase(const std::string &name)
        : name(name) {
    }
    MatrixBase()
        : MatrixBase("test") {
    }
    virtual std::string getName() const override {
        return name;
    }
    virtual std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }
    virtual FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::MatrixCPU::allocate",
            .args = {x, y, l_x, l_y},
        };
    }
    virtual FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "destroy",
            .args = {},
        };
    }
    virtual FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert",
            .args = {x, y, l_x, l_y},
        };
    }
    virtual FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, y, l_x, l_y},
        };
    }

    virtual std::string getType() const = 0;
};

class MatrixViewBasic : public MatrixBase {
public:
    MatrixViewBasic(const std::string &name)
        : MatrixBase(name) {
    }
    MatrixViewBasic()
        : MatrixBase("test") {
    }
    std::string getType() const override {
        return "gern::impl::MatrixViewBasic";
    }
};

class MatrixViewPackOnQuery : public MatrixBase {
public:
    MatrixViewPackOnQuery(const std::string &name)
        : MatrixBase(name) {
    }
    MatrixViewPackOnQuery()
        : MatrixBase("test") {
    }
    std::string getType() const override {
        return "gern::impl::MatrixViewPackOnQuery";
    }
    virtual FunctionSignature getQueryFunction() const override {
        return FunctionSignature{
            .name = "query",
            .args = {x, y},
            .template_args = {l_x, l_y}
        };
    }
    bool insertQuery() const override {
        return true;
    }
};

template <int64_t L_X, int64_t L_Y>
class MatrixTemplated : public MatrixBase {
public:
    MatrixTemplated(const std::string &name)
        : MatrixBase(name) {
    }
    MatrixTemplated()
        : MatrixBase("test") {
    }
    std::string getType() const override {
        return "gern::impl::MatrixTemplated<" + std::to_string(L_X) + ", " + std::to_string(L_Y) + ">";
    }
    
    bool freeAlloc() const override {
        return false;
    }
};


template <typename MatrixType>
class MatrixMultiplyCPU : public AbstractFunction {
public:
    MatrixMultiplyCPU()
        : A(new const MatrixType("A")),
          B(new const MatrixType("B")),
          C(new const MatrixType("C")) {
    }

    Annotation getAnnotation() override {
        Variable i("i");
        Variable j("j");
        Variable k("k");

        Variable ti("ti", Datatype::Int64);
        Variable tj("tj", Datatype::Int64);
        Variable tk("tk", Datatype::Int64);

        return annotate(For(i = Expr(0), ADTMember(C, "row", false), ti,
                            For(j = Expr(0), ADTMember(C, "col", false), tj,
                                Produces::Subset(C, {i, j, ti, tj}),
                                Consumes::Subsets(
                                    Reduce(k = Expr(0), k_dim, tk,
                                           SubsetObjMany({
                                               SubsetObj(A, {i, k, ti, tk}),
                                               SubsetObj(B, {k, j, tk, tj}),
                                           }))))));
    }

    FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::matrix_multiply";
        f.args = {Parameter(A), Parameter(B), Parameter(C), k_dim};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

protected:
    AbstractDataTypePtr A;
    AbstractDataTypePtr B;
    AbstractDataTypePtr C;
    Variable k_dim{"k_dim", Datatype::Int64};

    MatrixMultiplyCPU(AbstractDataTypePtr A, AbstractDataTypePtr B, AbstractDataTypePtr C)
        : A(A), B(B), C(C) {
    }
};


}  // namespace annot
}  // namespace gern