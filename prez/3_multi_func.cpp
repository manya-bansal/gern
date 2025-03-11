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
        // TODO
    });

    // ***** PROGRAM EVALUATION *****
    library::impl::ArrayCPU a(10);
    a.ascending();
    library::impl::ArrayCPU b(10);

    auto runner = compile_program(program);
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
