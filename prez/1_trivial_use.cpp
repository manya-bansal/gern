#include "gern_gen/program.cpp"
#include <cassert>
#include <iostream>

int main() {
    gern::impl::ArrayCPU a(10);
    a.ascending();
    gern::impl::ArrayCPU b(10);

    function_3(a, b);

    for (int i = 0; i < 10; i++) {
        std::cout << a.data[i] << " " << b.data[i] << std::endl;
        assert(a.data[i] + 1 == b.data[i]);
    }
}
