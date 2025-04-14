#pragma once

#include "gern_annot/adt.h"
#include "gern_annot/functions.h"

namespace schedules {

Composable kernel_1(int64_t m, int64_t n, int64_t k, int64_t block_size) {
    using AType = annot::MatrixGlobalToGlobal;
    using BType = annot::MatrixGlobalToGlobal;
    using CType = annot::MatrixGlobalToGlobal;

    auto A_DS = AbstractDataTypePtr(new const AType("A", m, k, block_size, false));
    auto B_DS = AbstractDataTypePtr(new const BType("B", k, n, block_size, false));
    auto C_DS = AbstractDataTypePtr(new const CType("C", m, n, block_size, false));

    Variable k_dim("k_dim");
    Variable block_x("block_x");
    Variable block_y("block_y");
    Variable thread_x("thread_x");
    Variable thread_y("thread_y");

    block_x = block_x.bind(32);   // 32 rows per block_x
    block_y = block_y.bind(32);   // 32 cols per block_y
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

    return program;
}
}  // namespace schedules