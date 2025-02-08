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

    annot::MatrixMultiply<K> matrix_multiply{A, B, C};

    Composable program = {
        matrix_multiply(A, B, C),
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