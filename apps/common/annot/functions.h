#pragma once

#include "annot/adt.h"

using namespace gern;

namespace annot {

class ExpMatrix : public AbstractFunction {
public:
    ExpMatrix(AbstractDataTypePtr input,
              AbstractDataTypePtr output)
        : input(input),
          output(output) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x", true);
        Variable l_y("l_y", true);

        return For(x = Expr(0), ADTMember(output, "row", true), l_x,
                   For(y = Expr(0), ADTMember(output, "col", true), l_y,
                       Produces::Subset(output, {x, y, l_x, l_y}),
                       Consumes::Subset(input, {x, y, l_x, l_y})))
            .occupies({Grid::Unit::SCALAR_UNIT});
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "exp_matrix";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "impl/gpu-matrix-const.h",
            "impl/impl.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

class MaxRow : public AbstractFunction {
public:
    MaxRow(AbstractDataTypePtr input,
           AbstractDataTypePtr output)
        : input(input), output(output) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x", true);
        Variable l_y("l_y", true);
        Variable col{"col", true};

        return For(x = Expr(0), ADTMember(output, "size", true), l_x,
                   Produces::Subset(output, {x, l_x}),
                   Consumes::Subsets(
                       Reduce(y = Expr(0), ADTMember(input, "col", true), l_y,
                              SubsetObjMany({
                                  SubsetObj(input, {x, y, l_x, col}),
                              })

                                  )))
            .occupies({Grid::Unit::SCALAR_UNIT});
    }

    std::vector<std::string> getHeader() override {
        return {
            "impl/gpu-matrix-const.h",
            "impl/impl.h",
        };
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "max_shuffle";
        f.args = {Parameter(output), Parameter(input)};
        f.template_args = {stride};
        return f;
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable stride{"stride", true};
};

class SumRow : public MaxRow {
public:
    SumRow(AbstractDataTypePtr input,
           AbstractDataTypePtr output)
        : MaxRow(input, output) {
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "sum_row";
        f.args = {Parameter(output), Parameter(input)};
        f.template_args = {stride};
        return f;
    }
};

class SubstractVec : public AbstractFunction {
public:
    SubstractVec(AbstractDataTypePtr vec,
                 AbstractDataTypePtr input,
                 AbstractDataTypePtr output)
        : vec(vec), input(input), output(output) {
    }

    Annotation getAnnotation() override {

        Variable x("x");
        Variable y("y");
        Variable l_x("l_x", true);
        Variable l_y("l_y", true);

        return annotate(For(x = Expr(0), ADTMember(output, "row", true), l_x,
                            For(y = Expr(0), ADTMember(output, "col", true), l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subsets(
                                    SubsetObjMany({
                                        SubsetObj(input, {x, y, l_x, l_y}),
                                        SubsetObj(vec, {x, l_x}),
                                    })))));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "subtract_vec";
        f.args = {Parameter(vec), Parameter(input), Parameter(output)};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "impl/gpu-matrix-const.h",
            "impl/impl.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    AbstractDataTypePtr vec;
};

class DivideVec : public SubstractVec {
public:
    DivideVec(AbstractDataTypePtr vec,
              AbstractDataTypePtr input,
              AbstractDataTypePtr output)
        : SubstractVec(vec, input, output) {
    }
    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "divide_vec";
        f.args = {Parameter(vec), Parameter(input), Parameter(output)};
        return f;
    }
};

class BlurX : public AbstractFunction {
public:
    BlurX(AbstractDataTypePtr input, AbstractDataTypePtr output)
        : input(input), output(output) {
    }
    Annotation getAnnotation() override {
        Variable x("x");
        Variable y("y");
        Variable l_x("l_x", true);
        Variable l_y("l_y", true);

        return annotate(For(x = Expr(0), ADTMember(output, "row", true), l_x,
                            For(y = Expr(0), ADTMember(output, "col", true), l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(
                                    input, {x, y, l_x, l_y + stride - 1}))));
    }
    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "blur_x";
        f.args = {Parameter(input), Parameter(output)};
        f.template_args = {stride};
        return f;
    }
    std::vector<std::string> getHeader() override {
        return {
            "impl/gpu-matrix-const.h",
            "impl/impl.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable stride{"stride", true};
};

}  // namespace annot