#include "cassert"
#include "cpu-array.h"

void function_5(gern::impl::ArrayCPU &input, gern::impl::ArrayCPU &output) {
    int64_t _gern_step_3 = output.size;
    int64_t _gern_x_4 = 0;

    int64_t _gern_x_2 = _gern_x_4;
    int64_t _gern_step_1 = _gern_step_3;
    gern::impl::ArrayCPU temp = gern::impl::ArrayCPU::allocate(_gern_x_2, _gern_step_1);

    gern::impl::add_1(input, temp);

    gern::impl::add_1(temp, output);

    temp.destroy();
}

extern "C" {
void hook_function_5(void **args) {
    gern::impl::ArrayCPU &input = *((gern::impl::ArrayCPU *)args[0]);
    gern::impl::ArrayCPU &output = *((gern::impl::ArrayCPU *)args[1]);
    function_5(input, output);
}
}
