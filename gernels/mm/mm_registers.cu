
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

    constexpr int64_t m = 8;
    constexpr int64_t n = 8;
    constexpr int64_t k = 8;
    constexpr int64_t tile_size_m = 4;
    constexpr int64_t tile_size_n = 8;
    constexpr int64_t tile_size_k = 4;
    constexpr int64_t block_size = 1;

    using AType = annot::MatrixGPU;
    using BType = annot::MatrixGPU;
    using CType = annot::MatrixGPU;

    using AImpl = impl::MatrixGPU<m, k, k, block_size>;
    using BImpl = impl::MatrixGPU<k, n, n, block_size>;
    using CImpl = impl::MatrixGPU<m, n, n, block_size>;

    auto A_DS = AbstractDataTypePtr(new const AType("A", m, k, block_size, false));
    auto B_DS = AbstractDataTypePtr(new const BType("B", k, n, block_size, false));
    auto C_DS = AbstractDataTypePtr(new const CType("C", m, n, block_size, false));

    Variable k_dim("k_dim");
    Variable c_row("c_row");
    Variable c_col("c_col");
    Variable k_tile("k_tile");

    c_row = c_row.bind(tile_size_m);
    c_col = c_col.bind(tile_size_n);
    k_tile = k_tile.bind(tile_size_k);
    k_dim = k_dim.bind(k);

    // Get the function object.
    annot::MatrixMultiplyReg mm(A_DS, B_DS, C_DS);
    auto mm_sp = &mm[{
        {"k_dim", k_dim},
    }];

    // Our program.
    Composable program = {
        Global(
            (Tile(C_DS["row"], c_row) || Grid::Unit::THREAD_X)(
                (Tile(C_DS["col"], c_col) || Grid::Unit::THREAD_Y)(
                    Reduce(k_dim, k_tile)(
                        (*mm_sp)(A_DS, B_DS, C_DS))))),
    };

    Runner::Options options;
    options.filename = "hello_mm.cu";
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

    runner.evaluate({{A_DS.getName(), &A},
                     {B_DS.getName(), &B},
                     {C_DS.getName(), &C}});

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
