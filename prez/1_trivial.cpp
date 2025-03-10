#include "helpers.h"

int main() {

    // Our first, simple program.
    auto a = mk_array("a");
    auto b = mk_array("b");

    gern::annot::add_1 add_1;

    // TODO

    Composable program = {
        add_1(a, b),
    };

    compile_program(program);
}
