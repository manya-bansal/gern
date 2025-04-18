#pragma once

#include "adt.h"
#include "annotations/abstract_function.h"
#include "compose/composable.h"

using namespace gern;

namespace annot {

class GridReduce : public AbstractFunction {
public:
    GridReduce()
        : input(new const FloatPtr("output")),
          output(new const ArrayGPU("input")) {
    }

    FunctionSignature getFunction() override {
        return FunctionSignature{
            .name = "grid_reduce_total",
            .args = {Parameter(output), Parameter(input)},
            .template_args = {k, block_size},
        };
    }

    Annotation getAnnotation() override {
        Variable i{"i"};
        return Computes(
                   Produces::Subset(output, {}),
                   Reducible(i = Expr(0), k, block_size,
                             SubsetObj(input, {i, block_size})))
            .occupies({Grid::Unit::THREAD_X})
            .assumes({Grid::Dim::BLOCK_DIM_X == block_size});
    }

    std::vector<std::string> getHeader() override {
        return {
            "wrappers/reduce_wrappers.cuh",
        };
    }

private:
    Variable k{"k", Datatype::Int64, true};
    Variable block_size{"block_size", Datatype::Int64, true};
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
};

class BlockReduce : public AbstractFunction {
public:
    BlockReduce()
        : input(new const ArrayGPU("output")),
          output(new const ArrayGPU("input")) {
    }

    FunctionSignature getFunction() override {
        return FunctionSignature{
            .name = "block_reduce",
            .args = {Parameter(output), Parameter(input)},
            .template_args = {block_size},
        };
    }

    Annotation getAnnotation() override {
        Variable i{"i"};
        Variable lx{"lx"};
        return Tileable(i = Expr(0), output["size"], lx,
                        Produces::Subset(output, {i, lx}),
                        Consumes::Subset(input, {i * block_size * lx,
                                                 block_size * lx}))
            .occupies({Grid::Unit::THREAD_X})
            .assumes({Grid::Dim::BLOCK_DIM_X == block_size});
    }

    std::vector<std::string> getHeader() override {
        return {
            "wrappers/reduce_wrappers.cuh",
        };
    }

private:
    Variable block_size{"block_size", Datatype::Int64, true};
    AbstractDataTypePtr input;
    AbstractDataTypePtr output;
};

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
        Variable tk{"tk", Datatype::Int64, true};
        return annotate(Computes(
            Produces::Subset(output, {}),
            Reducible(i = Expr(0), k, tk,
                      SubsetObj(input, {i, tk}))));
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