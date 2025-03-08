#include "helpers.h"
#include "library/array/impl/cpu-array.h"

using namespace gern;

int main() {

    // ***** PROGRAM DEFINITION *****
    // Our first, simple program.
    auto input = mk_array("input");
    auto temp = mk_array("temp");
    auto output = mk_array("output");
    Variable t("t");
    Variable t2("t2");
    gern::annot::add_1 add_1;

    Composable program({
        Tile(temp["size"], t)(
            Tile(temp["size"], t2)(
                add_1(input, temp))),
        Tile(output["size"], t)(
            add_1(temp, output)),
    });

    // ***** PROGRAM EVALUATION *****
    gern::impl::ArrayCPU a(10);
    a.ascending();
    gern::impl::ArrayCPU b(10);
    int64_t t_val = 2;
    int64_t t2_val = 1;

    auto runner = compile_program(program, "tile_program.cpp");
    runner.evaluate(
        {
            {"output", &b},
            {"input", &a},
            {"t", &t_val},
            {"t2", &t2_val},
        });

    std::cout << b << std::endl;
    // SANITY CHECK
    for (int i = 0; i < 10; i++) {
        assert(a.data[i] + 2 == b.data[i]);
    }
}
