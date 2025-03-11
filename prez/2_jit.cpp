#include "helpers.h"
#include "library/array/impl/cpu-array.h"

int main() {

    // ***** PROGRAM DEFINITION *****
    // Our first, simple program.
    auto a_def = mk_array("input");
    auto b_def = mk_array("output");

    gern::annot::add_1 add_1;

    Composable program = {
        add_1(a_def, b_def),
    };

    auto runner = compile_program(program);

    library::impl::ArrayCPU a(10);
    a.ascending();
    library::impl::ArrayCPU b(10);

    runner.evaluate({
        {"input", &a},
        {"output", &b},
    });

    for (int i = 0; i < 10; i++) {
        std::cout << "a[" << i << "] = " << a.data[i]
                  << ", b[" << i << "] = " << b.data[i] << std::endl;
        assert(a.data[i] + 1 == b.data[i]);
    }
}
