#include "cassert"
#include "cpu-array.h"

void function_3(library::impl::ArrayCPU &input,
                library::impl::ArrayCPU &output) {

    library::impl::add_1(input, output);
}

extern "C" {
void hook_function_3(void **args) {
    library::impl::ArrayCPU &input = *((library::impl::ArrayCPU *)args[0]);
    library::impl::ArrayCPU &output = *((library::impl::ArrayCPU *)args[1]);
    function_3(input, output);
}
}
