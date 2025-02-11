#include "annot/adt.h"
#include "annot/functions.h"
#include "annotations/abstract_function.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "impl/gpu-matrix-const.h"

#include <iostream>

using namespace gern;

int main() {
    constexpr int M = 16384;
    constexpr int N = 16384;
    constexpr int K = 16384;
    constexpr int row_major = true;

    using MatrixTypeA = impl::MatrixGPU<M, K, K, !row_major>;
    using Annot_MatrixTypeA = annot::MatrixGPU<M, K, K, !row_major>;
    using MatrixTypeB = impl::MatrixGPU<K, N, N, row_major>;
    using Annot_MatrixTypeB = annot::MatrixGPU<K, N, N, row_major>;
    using MatrixTypeC = impl::MatrixGPU<M, N, N, row_major>;
    using Annot_MatrixTypeC = annot::MatrixGPU<M, N, N, row_major>;

    auto A = AbstractDataTypePtr(new const Annot_MatrixTypeA("A", false));
    auto B = AbstractDataTypePtr(new const Annot_MatrixTypeB("B", false));
    auto C = AbstractDataTypePtr(new const Annot_MatrixTypeC("C", false));

    Variable tj{"tj"};
    Variable tj1{"tj1"};
    Variable tj2{"tj2"};

    Variable ti{"ti"};
    Variable ti1{"ti1"};
    Variable ti2{"ti2"};

    Variable tk{"tk"};
    Variable tk1{"tk1"};
    Variable tk2{"tk2"};

    Variable k("k");

    annot::MatrixMultiply matrix_multiply{A, B, C};

    // auto matrix_multiply_s = &matrix_multiply[{
    //     {"k", k.bindToInt64(8)},
    // }];

    // // For the inner mm
    // Composable program = {
    //     Global(
    //         Reduce(C["reduce"], tk.bind(16))(
    //             matrix_multiply(A, B, C)))};

    Runner::Options options;
    options.filename = "inner_mm.cu";
    options.include = "-I /home/manya/gern/apps/common"
                      " -I /home/manya/gern/test/";
    options.arch = "89";
    // FunctionPtr inner_mm(program, options);

    // The outer mm
    // Composable program = {
    //     Global(
    //         (Tile(C["row"], ti.bindToInt64(128)) || Grid::Unit::BLOCK_Y)(
    //             (Tile(C["col"], tj.bindToInt64(128)) || Grid::Unit::BLOCK_X)(
    //                 Reduce(C["reduce"], tk.bindToInt64(8))(
    //                     matrix_multiply(A, B, C)))))};

    // The middle mm
    Composable program = {
        Global(
            (Tile(C["row"], ti.bind(128)) || Grid::Unit::BLOCK_Y)(
                (Tile(C["col"], tj.bind(128)) || Grid::Unit::BLOCK_X)(
                    Reduce(C["reduce"], tk.bind(16))(
                        // Want to stage here
                        // Want to stage here
                        // Then continue tiling.
                        // (Call into MM (passing on bigger input than necessary))
                        Tile(C["row"], ti1.bind(64))(
                            Tile(C["col"], tj1.bind(64))(
                                (Tile(C["row"], ti2.bind(8)) || Grid::Unit::THREAD_Y)(
                                    (Tile(C["col"], tj2.bind(4)) || Grid::Unit::THREAD_X)(
                                        Reduce(C["reduce"], tk2.bind(1))(
                                            matrix_multiply(A, B, C)))))))))),
    };

    // // For the middle mm
    // Composable program = {
    //     // Additionally tile using warps.
    //     Global(
    //         (Tile(C["row"], ti.bindToInt64(8)) || Grid::Unit::THREAD_X)(
    //             (Tile(C["col"], tj.bindToInt64(4)) || Grid::Unit::THREAD_Y)(
    //                 matrix_multiply(A, B, C)))),
    // };

    Runner run(program);

    run.compile(options);

    // MatrixTypeA a;
    // a.ascending();
    // MatrixTypeB b;
    // b.ascending();

    return 0;
}