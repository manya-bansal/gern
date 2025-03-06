#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_blur.h"
#include "value.h"
#include <cuda_runtime.h>

#define SAMPLES 10
#define ITERATIONS 10

int main(int argc, char **argv) {
    // for (int num_elem = SLIDING_START_DIM; num_elem <= SLIDING_END_DIM; num_elem += SLIDING_STEP_DIM) {
    int num_elem = size_matrix + 2;

    Halide::Runtime::Buffer<float> input(num_elem, num_elem);
    Halide::Runtime::Buffer<float> blur_y(num_elem - 2, num_elem - 2);

    int i = 0;
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            input(x, y) = i++;
        }
    }

    // double time = Halide::Tools::benchmark(
    //     SAMPLES, ITERATIONS, [&]() { halide_blur(input, blur_y); });

    for (int i = 0; i < 10; i++) {
        halide_blur(input, blur_y);
    }

    cudaDeviceSynchronize();
    // Start actual benchmarking
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++) {
        halide_blur(input, blur_y);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = (end - start);
    auto time = (duration.count() / 200) / 1e6;

    double bytes = (num_elem - 2) * (num_elem - 2) * 4 * 2;

    std::cout << num_elem - 2 << std::endl;
    std::cout << bytes / time << std::endl;

    blur_y.copy_to_host();

    // for (int y = 0; y < blur_y.height(); ++y) {
    //     for (int x = 0; x < blur_y.width(); ++x) {
    //         // std::cout << blur_y(x, y) << " ";
    //     }
    // std::cout << std::endl;
    // }

    return 0;
}