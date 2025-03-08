#include "helpers.h"
#include "library/array/impl/cpu-array.h"

int main() {

    // ***** PROGRAM DEFINITION *****
    // Our first, simple program.
    auto input = mk_array("input");
    auto temp = mk_array("temp");
    auto output = mk_array("output");

    gern::annot::add_1 add_1;

    Composable program({
        add_1(input, temp),
        add_1(temp, output),
    });

    // ***** PROGRAM EVALUATION *****
    gern::impl::ArrayCPU a(10);
    a.ascending();
    gern::impl::ArrayCPU b(10);

    auto runner = compile_program(program, "multi_func.cpp");
    runner.evaluate(
        {
            {"output", &b},
            {"input", &a},
        });

    // SANITY CHECK
    for (int i = 0; i < 10; i++) {
        assert(a.data[i] + 2 == b.data[i]);
    }
}
