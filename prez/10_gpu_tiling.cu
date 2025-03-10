#include "helpers.h"
#include "library/array/annot/gpu-array.h"
#include "library/array/impl/gpu-array.h"

using namespace gern;

int main() {
    // ***** PROGRAM DEFINITION *****
    auto a = mk_array_gpu("a");
    auto b = mk_array_gpu("b");

    gern::annot::Add1GPU add_1;
    Variable len("len");
    Variable tile("tile");
    Variable tile2("tile2");

    Composable program({
        Global(
            (Tile(b["size"], tile.bind(5)))(
                (Tile(b["size"], tile2.bind(1)))(
                    add_1(a, b)))),
    });

    // ***** PROGRAM EVALUATION *****
    library::impl::ArrayGPU a_gpu(10);
    a_gpu.ascending();
    library::impl::ArrayGPU b_gpu(10);
    b_gpu.ascending();

    auto runner = compile_program_gpu(program);
    runner.evaluate({
        {"a", &a_gpu},
        {"b", &b_gpu},
    });

    auto b_cpu = b_gpu.get();
    auto a_cpu = a_gpu.get();

    for (int i = 0; i < 10; i++) {
        assert(b_cpu.data[i] == a_cpu.data[i] + 1);
    }
}