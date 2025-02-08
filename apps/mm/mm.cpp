#include "annot/adt.h"
#include "annot/functions.h"
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
    Variable ti{"ti"};
    Variable tk{"tk"};
    Variable tk2{"tk2"};
    Variable k("k");

    annot::MatrixMultiply matrix_multiply{A, B, C};

    auto matrix_multiply_s = &matrix_multiply[{
        {"k", k.bindToInt64(8)},
    }];

    // For the inner mm
    // Composable program = {
    //     Global(
    //         Reduce(C["reduce"], tk.bindToInt64(16))(
    //             matrix_multiply(A, B, C)))};

    // The outer mm
    // Composable program = {
    //     Global(
    //         (Tile(C["row"], ti.bindToInt64(128)) || Grid::Unit::BLOCK_Y)(
    //             (Tile(C["col"], tj.bindToInt64(128)) || Grid::Unit::BLOCK_X)(
    //                 Reduce(C["reduce"], tk.bindToInt64(16))(
    //                     matrix_multiply(A, B, C)))))};

    // For the middle mm
    Composable program = {
        // Additionally tile using warps.
        Global(
            (Tile(C["row"], ti.bindToInt64(8)) || Grid::Unit::THREAD_X)(
                (Tile(C["col"], tj.bindToInt64(4)) || Grid::Unit::THREAD_Y)(
                    matrix_multiply(A, B, C)))),
    };

    Runner run(program);
    Runner::Options options;
    options.include = "-I /home/manya/gern/apps/common"
                      " -I /home/manya/gern/test/";
    options.arch = "89";
    run.compile(options);

    // MatrixTypeA a;
    // a.ascending();
    // MatrixTypeB b;
    // b.ascending();

    return 0;
}