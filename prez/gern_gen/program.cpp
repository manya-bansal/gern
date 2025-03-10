#include "cassert"
#include "cpu-array.h"

void function_3(library::impl::ArrayCPU &a, library::impl::ArrayCPU &b) {

    library::impl::add_1(a, b);
}

extern "C" {
void hook_function_3(void **args) {
    library::impl::ArrayCPU &a = *((library::impl::ArrayCPU *)args[0]);
    library::impl::ArrayCPU &b = *((library::impl::ArrayCPU *)args[1]);
    function_3(a, b);
}
}
