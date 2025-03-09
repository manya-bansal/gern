#include "helpers.h"
#include "library/array/annot/cpu-array.h"
#include "library/array/impl/cpu-array.h"

using namespace gern;

int main() {
    // ***** PROGRAM DEFINITION *****
    auto input = mk_array("input");
    auto output = mk_array("output");

    gern::annot::reduction reduce;
    Variable len("len");
    Variable tile_size("tile_size");

    Composable program({
        Reduce(len, tile_size)(
            reduce(input, output, len)),
    });

    // ***** PROGRAM EVALUATION *****
    gern::impl::ArrayCPU a(10);
    a.ascending();
    gern::impl::ArrayCPU b(10);
    int64_t len_val = 10;
    int64_t tile_size_val = 2;

    auto runner = compile_program(program);
    runner.evaluate({
        {"input", &a},
        {"output", &b},
        {"len", &len_val},
        {"tile_size", &tile_size_val},
    });

    // ***** SANITY CHECK *****
    for (int i = 0; i < 10; i++) {
        assert(b.data[i] == (len_val * (len_val - 1)) / 2);
    }
}
