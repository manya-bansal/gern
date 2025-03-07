#include "../current_path.h"
#include "compose/runner.h"
#include "gern_annot/adt.h"
#include "gern_annot/functions.h"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {

    constexpr int64_t m = 64;
    constexpr int64_t n = 64;
    constexpr int64_t k = 8;
    constexpr int64_t block_size = 1;

    using AType = annot::MatrixGlobalToGlobal<m, k, block_size>;
    using BType = annot::MatrixGlobalToGlobal<k, n, block_size>;
    using CType = annot::MatrixGlobalToGlobal<m, n, block_size>;

    using AImpl = impl::MatrixGPU<m, k, k, block_size>;
    using BImpl = impl::MatrixGPU<k, n, n, block_size>;
    using CImpl = impl::MatrixGPU<m, n, n, block_size>;

    auto A_DS = AbstractDataTypePtr(new const AType("A", false));
    auto B_DS = AbstractDataTypePtr(new const BType("B", false));
    auto C_DS = AbstractDataTypePtr(new const CType("C", false));

    Variable k_dim("k_dim");
    Variable block_x("block_x");
    Variable block_y("block_y");
    Variable thread_x("thread_x");
    Variable thread_y("thread_y");

    block_x = block_x.bind(32);   // 8 elements per block_x
    block_y = block_y.bind(32);   // 8 elements per block_y
    thread_x = thread_x.bind(1);  // 1 element per thread_x
    thread_y = thread_y.bind(1);  // 1 element per thread_y
    k_dim = k_dim.bind(k);

    annot::MatrixMultiply mm(A_DS, B_DS, C_DS);
    auto mm_sp = &mm[{
        {"k_dim", k_dim},
    }];

    // Distribute over blocks and threads trivially.
    Composable program = {
        Global(
            (Tile(C_DS["row"], block_x) || Grid::Unit::BLOCK_X)(
                (Tile(C_DS["col"], block_y) || Grid::Unit::BLOCK_Y)(
                    (Tile(C_DS["row"], thread_x) || Grid::Unit::THREAD_X)(
                        (Tile(C_DS["col"], thread_x) || Grid::Unit::THREAD_Y)(
                            (*mm_sp)(A_DS, B_DS, C_DS)))))),
    };

    Runner::Options options;
    options.filename = "kernel_1.cu";
    options.cpp_std = "c++17";
    options.arch = GERNELS_ARCH;
    options.include = " -I" + std::string(GERNELS_PATH) + "/mm";

    Runner runner(program);
    runner.compile(options);

    AImpl A;
    A.ascending();
    BImpl B;
    B.ascending();
    CImpl C;
    C.vvals(0.0f);

    // Set up all the values.
    runner.evaluate({{A_DS.getName(), &A},
                     {B_DS.getName(), &B},
                     {C_DS.getName(), &C}});

    // Make sure the output is correct!
    auto C_cpu = C.get();
    auto C_cpu_ref = C.get();
    C_cpu_ref.vvals(0.0f);
    auto A_cpu = A.get();
    auto B_cpu = B.get();
    matrix_multiply_cpu(A_cpu, B_cpu, C_cpu_ref);

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            assert(std::abs(C_cpu(i, j) - C_cpu_ref(i, j)) < 1e-5);
        }
    }

    std::cout << "Success!" << std::endl;

    // Free everything.
    C_cpu.destroy();
    C_cpu_ref.destroy();
    A_cpu.destroy();
    B_cpu.destroy();
    A.destroy();
    B.destroy();
    C.destroy();
}