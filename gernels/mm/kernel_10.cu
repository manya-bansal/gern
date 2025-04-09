#include "../current_path.h"
#include "compose/runner.h"
#include "float-error.h"
#include "gern_annot/adt.h"
#include "gern_annot/functions.h"
#include "gern_annot/shmem_interface.h"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {

    constexpr int64_t m = 8 * 8 * 8 * 2;
    constexpr int64_t n = 8 * 8 * 8 * 2;
    constexpr int64_t k = 8 * 8 * 8 * 2;
    constexpr int64_t block_size = 1;

    using AType = annot::MatrixGlobalToSharedFlat<m, k, block_size>;
    using BType = annot::MatrixGlobalToSharedFlat<k, n, block_size>;
    using CType = annot::MatrixGlobalToGlobal<m, n, block_size>;

    using CTypeReg = annot::MatrixQueryRegNoVector<m, n, block_size>;

    using AImpl = impl::MatrixGPU<m, k, k, block_size>;
    using BImpl = impl::MatrixGPU<k, n, n, block_size>;
    using CImpl = impl::MatrixGPU<m, n, n, block_size>;

    auto A_DS = AbstractDataTypePtr(new const AType("A", false));
    auto B_DS = AbstractDataTypePtr(new const BType("B", false));
    auto C_DS = AbstractDataTypePtr(new const CType("C", false));

    auto obj = AType("A_vec", false);
    auto obj_reg = CTypeReg("C_reg", false);

    Variable k_dim("k_dim");
    Variable k_tiled("k_tiled");

    Variable block_x("block_x");
    Variable block_y("block_y");
    Variable thread_x("thread_x");
    Variable thread_y("thread_y");
    Variable thread_k("thread_k");
    Variable warp_x("warp_x");
    Variable warp_y("warp_y");
    Variable smem_size("smem_size");
    Variable one_val("one_val");

    constexpr int64_t BM = 8 * 4;
    constexpr int64_t BN = 8 * 4;
    constexpr int64_t BK = 8 * 8;

    constexpr int64_t WM = 32;
    constexpr int64_t WN = 32;
    constexpr int64_t WK = 32;

    constexpr int64_t TM = 1;
    constexpr int64_t TN = 1;
    constexpr int64_t TK = 8;

    constexpr int64_t BLOCK_DIM_X = (BM / WM) * 32;
    constexpr int64_t BLOCK_DIM_Y = (BN / WN) * 32;

    block_x = block_x.bind(BM);
    block_y = block_y.bind(BN);

    warp_x = warp_x.bind(WM);
    warp_y = warp_y.bind(WN);

    thread_x = thread_x.bind(TM);
    thread_y = thread_y.bind(TN);
    thread_k = thread_k.bind(TK);

    k_dim = k_dim.bind(k);

    k_tiled = k_tiled.bind(BK);

    int64_t smem_size_val = 32 * 32 * 8 * 4;  // overallocating by a bit

    annot::MatrixMultiply mm(A_DS, B_DS, C_DS);
    auto mm_sp = &mm[{
        {"k_dim", k_dim},
    }];

    // Some critical assumptions.

    // Use because we stage data into smem using threads.
    static_assert((BM * BK) >= (BLOCK_DIM_X * BLOCK_DIM_Y), "BM * BK must be greater than or equal to BLOCK_DIM_X * BLOCK_DIM_Y");
    static_assert((BM * BK) % (BLOCK_DIM_X * BLOCK_DIM_Y) == 0, "BM * BK must be divisible by BLOCK_DIM_X * BLOCK_DIM_Y");

    static_assert((BK * BN) % (BLOCK_DIM_X * BLOCK_DIM_Y) == 0, "BK * BN must be divisible by BLOCK_DIM_X * BLOCK_DIM_Y");
    static_assert((BK * BN) % (BLOCK_DIM_X * BLOCK_DIM_Y) == 0, "BK * BN must be divisible by BLOCK_DIM_X * BLOCK_DIM_Y");

    static_assert((WN / TN) == 32, "WN must be divisible by TN");
    static_assert((WN % TN) == 0, "WN must be divisible by TN");

    static_assert((WM / TM) == 32, "WM must be divisible by TM");
    static_assert((WM % TM) == 0, "WM must be divisible by TM");

    // GPU specific constraint.
    static_assert((BLOCK_DIM_X * BLOCK_DIM_Y) <= 1024, "BLOCK_DIM_X * BLOCK_DIM_Y must be less than 1024");
    static_assert((BLOCK_DIM_X * BLOCK_DIM_Y) % 32 == 0, "BLOCK_DIM_X * BLOCK_DIM_Y must be divisible by 32");

    // static_assert((WM * WN) <= 1024, "BLOCK_DIM_X * BLOCK_DIM_Y must be less than 1024");

    Composable program = {
        Global(
            (Tile(C_DS["row"], block_x) || Grid::Unit::BLOCK_X)(
                (Tile(C_DS["col"], block_y) || Grid::Unit::BLOCK_Y)(
                    (Reduce(k_dim, k_tiled))(
                        Stage(A_DS,
                              Stage(B_DS,
                                    (Tile(C_DS["row"], warp_x) || Grid::Unit::WARP_Y)(
                                        (Tile(C_DS["col"], warp_y) || Grid::Unit::WARP_X)(

                                            (Tile(C_DS["row"], thread_x) || Grid::Unit::THREAD_X_IN_WRAPS)(
                                                (Tile(C_DS["col"], thread_y) || Grid::Unit::THREAD_Y_IN_WRAPS)(

                                                    Stage(C_DS, obj_reg.getQueryFunction(), obj_reg.getInsertFunction(),
                                                          Stage(A_DS, obj_reg.getQueryFunction(),
                                                                Stage(B_DS, obj_reg.getQueryFunction(),

                                                                      Reduce(k_dim, thread_k)(

                                                                          Stage(A_DS, obj.getView(),
                                                                                Stage(B_DS, obj.getView(),
                                                                                      (*mm_sp)(A_DS, B_DS, C_DS)))))))))))))))),
            {}, smem_size, TrivialManager(smem_size)),
    };

    Runner::Options options;
    options.filename = "kernel_10.cu";
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
                     {C_DS.getName(), &C},
                     {smem_size.getName(), &smem_size_val}});

    cudaDeviceSynchronize();

    // Make sure the output is correct!
    auto C_cpu = C.get();
    auto C_cpu_ref = C.get();
    C_cpu_ref.vvals(0.0f);
    auto A_cpu = A.get();
    auto B_cpu = B.get();
    matrix_multiply_cpu(A_cpu, B_cpu, C_cpu_ref);

    // std::cout << "C_cpu: " << C_cpu << std::endl;
    // std::cout << "C_cpu_ref: " << C_cpu_ref << std::endl;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            assert_close(C_cpu(i, j), C_cpu_ref(i, j));
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