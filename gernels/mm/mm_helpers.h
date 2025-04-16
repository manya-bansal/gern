#pragma once

#include "compose/runner.h"
#include "impl/matrix_multiply.h"

using namespace gern;

namespace mm_helpers {

#define tolerance 1e-3

#define assert_close(a, b) assert(std::abs(a - b) < tolerance)

Runner runner(Composable program, const std::string &filename) {
    Runner::Options options;
    options.filename = filename;
    options.cpp_std = "c++17";
    options.arch = GERNELS_ARCH;
    options.include = "-O3 -I" + std::string(GERNELS_PATH) + "/mm";
    Runner runner(program);
    runner.compile(options);
    return runner;
}

template<typename AImpl, typename BImpl, typename CImpl>
void evalute_and_check(Runner runner, AImpl &A, BImpl &B, CImpl &C) {
    A.ascending();
    B.ascending();
    C.vvals(0.0f);
    // Set up all the values.
    runner.evaluate({{"A", &A},
                     {"B", &B},
                     {"C", &C}});

    // Make sure the output is correct!
    auto C_cpu = C.get();
    auto C_cpu_ref = C.get();
    C_cpu_ref.vvals(0.0f);
    auto A_cpu = A.get();
    auto B_cpu = B.get();
    matrix_multiply_cpu(A_cpu, B_cpu, C_cpu_ref);

    for (int64_t i = 0; i < A.row; i++) {
        for (int64_t j = 0; j < B.col; j++) {
            assert_close(C_cpu(i, j), C_cpu_ref(i, j));
        }
    }

    std::cout << "Success!" << std::endl;

    C_cpu.destroy();
    C_cpu_ref.destroy();
    A_cpu.destroy();
    B_cpu.destroy();
}

// template<typename AImpl, typename BImpl, typename CImpl>
// void evalute_and_check_smem(Runner runner, AImpl &A, BImpl &B, CImpl &C) {
//     A.ascending();
//     B.ascending();
//     C.vvals(0.0f);
//     // Set up all the values.
//     runner.evaluate({{"A", &A},
//                      {"B", &B},
//                      {"C", &C}});

//     // Make sure the output is correct!
//     auto C_cpu = C.get();
//     auto C_cpu_ref = C.get();
//     C_cpu_ref.vvals(0.0f);
//     auto A_cpu = A.get();
//     auto B_cpu = B.get();
//     matrix_multiply_cpu(A_cpu, B_cpu, C_cpu_ref);

//     for (int64_t i = 0; i < A.row; i++) {
//         for (int64_t j = 0; j < B.col; j++) {
//             assert_close(C_cpu(i, j), C_cpu_ref(i, j));
//         }
//     }

//     std::cout << "Success!" << std::endl;

//     C_cpu.destroy();
//     C_cpu_ref.destroy();
//     A_cpu.destroy();
//     B_cpu.destroy();
// }

double gflops(int64_t m, int64_t n, int64_t k, double time) {
    double flops = 2.0 * m * n * k;
    // Divide by 1e9 to convert to GFLOPS.
    double gflops = flops / (time * 1e9);
    return gflops;
}
}  // namespace mm_helpers
