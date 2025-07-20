#pragma once

#include "annotations/abstract_function.h"
#include "config.h"
#include "library/array/annot/cpu-array.h"
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

    std::vector<Variable> getFields() const override {
        return {x, y, l_x, l_y};
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::MatrixCPU::allocate",
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

    bool insertQuery() const override {
        return true;
    }

protected:
    std::string name;
    Variable x{"x"};
    Variable y{"y"};
    Variable l_x{"l_x"};
    Variable l_y{"l_y"};
};

class MatrixCPUZero : public MatrixCPU {
public:
    MatrixCPUZero(const std::string &name)
        : MatrixCPU(name) {
    }

    FunctionSignature getAllocateFunction() const override {
        return FunctionSignature{
            .name = "gern::impl::MatrixCPU::allocate_zero",
            .args = {x, y, l_x, l_y},
        };
    }
};

class MatrixAddCPU : public AbstractFunction {
public:
    MatrixAddCPU()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::add";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(Tileable(x = Expr(0), output["row"], l_x,
                                 Tileable(y = Expr(0), output["col"], l_y,
                                          Produces::Subset(output, {x, y, l_x, l_y}),
                                          Consumes::Subset(input, {x, y, l_x, l_y}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::add";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MatrixTransposeCPU : public AbstractFunction {
public:
    MatrixTransposeCPU()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")) {
    }
    std::string getName() {
        return "gern::impl::transpose";
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(Tileable(x = Expr(0), output["row"], l_x,
                                 Tileable(y = Expr(0), output["col"], l_y,
                                          Produces::Subset(output, {x, y, l_x, l_y}),
                                          Consumes::Subset(input, {y, x, l_y, l_x}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::transpose";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};

};

class SumRow : public AbstractFunction {
public:
    SumRow()
        : input(new const MatrixCPU("input")),
          output(new const ArrayCPU("output")) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(Tileable(x = Expr(0), output["size"], l_x,
                                 Produces::Subset(output, {x, l_x}),
                                 Consumes::Subset(input, {x, 0, l_x, col})));
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::sum_row";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MaxRow : public SumRow {
public:
    MaxRow()
        : SumRow() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::max_row";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }
};

class ExpMatrix : public MatrixAddCPU {
public:
    ExpMatrix()
        : MatrixAddCPU() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::exp_matrix";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }
};

class SubtractVec : public AbstractFunction {
public:
    SubtractVec()
        : input(new const MatrixCPU("input")),
          output(new const MatrixCPU("output")),
          vec(new const ArrayCPU("vec")) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x");
        Variable l_y("l_y");

        Variable row("row");
        Variable col("col");

        return annotate(Tileable(x = Expr(0), output["row"], l_x,
                                 Tileable(y = Expr(0), output["row"], l_y,
                                          Produces::Subset(output, {x, y, l_x, l_y}),
                                          Consumes::Subsets(
                                              SubsetObjMany({
                                                  SubsetObj(input, {x, y, l_x, l_y}),
                                                  SubsetObj(vec, {x, l_x}),
                                              })))));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::subtract_vec";
        f.args = {Parameter(vec), Parameter(input), Parameter(output)};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "cpu-matrix.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    AbstractDataTypePtr vec;
    Variable end{"end"};
};

class DivideVec : public SubtractVec {
public:
    DivideVec()
        : SubtractVec() {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "gern::impl::divide_vec";
        f.args = {Parameter(vec), Parameter(input), Parameter(output)};
        return f;
    }
};

class MatrixMultiplyCPU : public AbstractFunction {
public:
    MatrixMultiplyCPU()
        : A(new const MatrixCPU("A")),
          B(new const MatrixCPU("B")),
          C(new const MatrixCPU("C")) {
    }

    Annotation getAnnotation() override {
        Variable i("i");
        Variable j("j");
        Variable k("k");

        Variable ti("ti", Datatype::Int64);
        Variable tj("tj", Datatype::Int64);
        Variable tk("tk", Datatype::Int64);

        return annotate(Tileable(i = Expr(0), ADTMember(C, "row", false), ti,
                                 Tileable(j = Expr(0), ADTMember(C, "col", false), tj,
                                          Produces::Subset(C, {i, j, ti, tj}),
                                          Consumes::Subsets(
                                              Reducible(k = Expr(0), k_dim, tk,
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

private:
    AbstractDataTypePtr A;
    AbstractDataTypePtr B;
    AbstractDataTypePtr C;
    Variable k_dim{"k_dim", Datatype::Int64};
};

}  // namespace annot
}  // namespace gern