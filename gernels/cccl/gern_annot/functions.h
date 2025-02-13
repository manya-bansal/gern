#pragma once

#include "adt.h"
#include "annotations/abstract_function.h"
#include "compose/composable.h"

using namespace gern;

namespace annot {

class GlobalSum : public AbstractFunction {
public:
    GlobalSum()
        : input(new const FloatPtr("output")),
          output(new const ArrayGPU("input")) {
    }

    FunctionSignature getFunction() override {
        return FunctionSignature{
            .name = "global_sum",
            .args = {Parameter(output), Parameter(input)},
            .template_args = {k},
        };
    }

    Annotation getAnnotation() override {
        Variable i{"i"};
        return annotate(Computes(
            Produces::Subset(output, {}),
            Reduce(i = Expr(0), input["size"], k,
                   SubsetObj(input, {i, k}))));
    }

    std::vector<std::string> getHeader() override {
        return {
            "wrappers/reduce_wrappers.cuh",
        };
    }

private:
    Variable k{"k", Datatype::Int64, true};
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
};

}  // namespace annot