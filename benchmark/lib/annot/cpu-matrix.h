#pragma once

#include "annotations/abstract_function.h"
#include <iostream>

namespace gern {
namespace annot {

class Matrix : public AbstractDataType {
private:
    std::string implementation;
    bool is_query_templated;

    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};

public:
    Matrix(const std::string &implementation, bool is_query_templated, const std::string &name)
        : implementation(implementation), is_query_templated(is_query_templated), name(name) {
    }

    virtual std::string getName() const override {
        return name;
    }
    virtual std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }
    virtual FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::" + implementation + "::allocate",
            .args = {x, y, l_x, l_y},
        };
    }
    virtual FunctionSignature getFreeFunction() const override {
        return FunctionSignature{
            .name = "destroy",
            .args = {},
        };
    }

    virtual FunctionSignature getQueryFunction() const override {
        return is_query_templated ?
                   FunctionSignature{
                       .name = "query",
                       .args = {x, y},
                       .template_args = {l_x, l_y}} :
                   FunctionSignature{
                       .name = "query",
                       .args = {x, y, l_x, l_y}};
    }

    virtual std::string getType() const override {
        return "gern::impl::" + implementation;
    }
    virtual FunctionSignature getInsertFunction() const override {
        return FunctionSignature{
            .name = "insert",
            .args = {x, y, l_x, l_y}};
    }
};

class MatrixMultiply : public AbstractFunction {
public:
    MatrixMultiply(const std::string function_name, AbstractDataTypePtr A, AbstractDataTypePtr B, AbstractDataTypePtr C)
        : function_name(function_name), A(A), B(B), C(C) {
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
        f.name = "gern::impl::" + function_name;
        f.args = {Parameter(A), Parameter(B), Parameter(C), k_dim};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

protected:
    std::string function_name;
    AbstractDataTypePtr A;
    AbstractDataTypePtr B;
    AbstractDataTypePtr C;
    Variable k_dim{"k_dim", Datatype::Int64};
};

}  // namespace annot
}  // namespace gern