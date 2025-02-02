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
        Variable l_x("l_x");
        Variable l_y("l_y");

        return annotate(For(x = Expr(0), output["row"], l_x,
                            For(y = Expr(0), output["col"], l_y,
                                Produces::Subset(output, {x, y, l_x, l_y}),
                                Consumes::Subset(input, {x, y, l_x, l_y}))));
    }

    virtual FunctionSignature getFunction() override {
        FunctionSignature f;
        f.name = "exp_matrix";
        f.args = {Parameter(input), Parameter(output)};
        return f;
    }

    std::vector<std::string> getHeader() override {
        return {
            "softmax/gpu-matrix-const.h",
            "softmax/impl.h",
        };
    }

protected:
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
    Variable end{"end"};
};

}  // namespace annot