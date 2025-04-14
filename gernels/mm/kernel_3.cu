#include "../benchmark.h"
#include "../current_path.h"
#include "compose/runner.h"
#include "float-error.h"
#include "gern_annot/adt.h"
#include "gern_annot/functions.h"
#include "gern_annot/shmem_interface.h"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include "kernel_3.h"
#include "mm_helpers.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {

    constexpr int64_t m = M_CONST;
    constexpr int64_t n = N_CONST;
    constexpr int64_t k = K_CONST;
    constexpr int64_t block_size = 1;

    using AType = annot::MatrixGlobalToShared;
    using BType = annot::MatrixGlobalToShared;
    using CType = annot::MatrixGlobalToGlobal;

    using AImpl = impl::MatrixGPU<m, k, k, block_size>;
    using BImpl = impl::MatrixGPU<k, n, n, block_size>;
    using CImpl = impl::MatrixGPU<m, n, n, block_size>;

    auto A_DS = AbstractDataTypePtr(new const AType("A", m, k, block_size, false));
    auto B_DS = AbstractDataTypePtr(new const BType("B", k, n, block_size, false));
    auto C_DS = AbstractDataTypePtr(new const CType("C", m, n, block_size, false));

    auto obj = AType("A_vec", m, k, block_size, false);

    Variable k_dim("k_dim");
    Variable k_tiled("k_tiled");

    Variable block_x("block_x");
    Variable block_y("block_y");
    Variable thread_x("thread_x");
    Variable thread_y("thread_y");
    Variable smem_size("smem_size");
    Variable one_val("one_val");

    block_x = block_x.bind(32);
    block_y = block_y.bind(32);
    thread_x = thread_x.bind(1);
    thread_y = thread_y.bind(1);
    k_dim = k_dim.bind(k);
    k_tiled = k_tiled.bind(32);
    int64_t smem_size_val = 32 * 32 * 8 * 2 + 1000;  // overallocating by a bit

    annot::MatrixMultiply mm(A_DS, B_DS, C_DS);
    auto mm_sp = &mm[{
        {"k_dim", k_dim},
    }];

    // Distribute over blocks and threads trivially.
    // But does memory coalescing flipping col and row basically gives memory coalescing.
    Composable program = {
        Global(
            (Tile(C_DS["row"], block_x) || Grid::Unit::BLOCK_X)(
                (Tile(C_DS["col"], block_y) || Grid::Unit::BLOCK_Y)(
                    (Reduce(k_dim, k_tiled))(
                        Stage(A_DS,
                              Stage(B_DS,
                                    (Tile(C_DS["row"], thread_x) || Grid::Unit::THREAD_Y)(
                                        (Tile(C_DS["col"], thread_y) || Grid::Unit::THREAD_X)(
                                            Stage(A_DS, obj.getView(),
                                                  Stage(B_DS, obj.getView(),
                                                        (*mm_sp)(A_DS, B_DS, C_DS)))))))))),
            {}, smem_size, TrivialManager(smem_size)),
    };

    Runner runner = mm_helpers::runner(program, "kernel_3.cu");

    AImpl A;
    A.ascending();
    BImpl B;
    B.ascending();
    CImpl C;
    C.vvals(0.0f);

    // mm_helpers::evalute_and_check(runner, A, B, C);
    auto func = [&]() {
        runner.evaluate({{A_DS.getName(), &A},
                         {B_DS.getName(), &B},
                         {C_DS.getName(), &C},
                         {smem_size.getName(), &smem_size_val}});
    };
    double time = benchmark::benchmark(10, 1, func, 2);
    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "GFLOPS: " << mm_helpers::gflops(m, n, k, time) << std::endl;
    std::cout << "% of peak " << mm_helpers::gflops(m, n, k, time) / (44 * 10) << std::endl;

    A.destroy();
    B.destroy();
    C.destroy();
}