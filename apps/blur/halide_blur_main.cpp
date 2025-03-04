#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "HalideBuffer.h"
#include "bench_range.h"
#include "halide_benchmark.h"
#include "halide_blur.h"

int main(int argc, char **argv) {
    // for (int num_elem = SLIDING_START_DIM; num_elem <= SLIDING_END_DIM; num_elem += SLIDING_STEP_DIM) {
    int num_elem = 4488 * 8 + 2;
    Halide::Runtime::Buffer<float> input(num_elem, num_elem);
    // Halide::Runtime::Buffer<float> blur_x(num_elem, num_elem - 2);
    Halide::Runtime::Buffer<float> blur_y(num_elem - 2, num_elem - 2);

    // for (int y = 0; y < input.height(); ++y) {
    //     for (int x = 0; x < input.width(); ++x) {
    //         input(x, y) = x + y;
    //     }
    // }

    // double time = Halide::Tools::benchmark(
    //     SAMPLES, ITERATIONS, [&]() { halide_blur(input, blur_y); });

    // std::cout << num_elem << std::endl;
    // std::cout << time << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    halide_blur(input, blur_y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "time taken: " << elapsed.count() << "s" << std::endl;

    auto start_2 = std::chrono::high_resolution_clock::now();
    halide_blur(input, blur_y);
    auto end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_2 = end_2 - start_2;
    std::cout << "time taken: " << elapsed_2.count() << "s" << std::endl;

    auto start_3 = std::chrono::high_resolution_clock::now();
    halide_blur(input, blur_y);
    auto end_3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_3 = end_3 - start_3;
    std::cout << "time taken: " << elapsed_3.count() << "s" << std::endl;
    // }

    return 0;
}