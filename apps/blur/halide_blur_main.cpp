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
    for (int num_elem = SLIDING_START_DIM; num_elem <= SLIDING_END_DIM; num_elem += SLIDING_STEP_DIM) {
        Halide::Runtime::Buffer<float> input(num_elem, num_elem);
        Halide::Runtime::Buffer<float> blur_x(num_elem, num_elem - 2);
        Halide::Runtime::Buffer<float> blur_y(num_elem - 2, num_elem - 2);

        for (int y = 0; y < input.height(); ++y) {
            for (int x = 0; x < input.width(); ++x) {
                input(x, y) = x + y;
            }
        }

        double time = Halide::Tools::benchmark(
            SAMPLES, ITERATIONS, [&]() { halide_blur(input, blur_y); });

        std::cout << num_elem << std::endl;
        std::cout << time << std::endl;
    }

    return 0;
}