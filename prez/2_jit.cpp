#include "helpers.h"
#include "library/array/impl/cpu-array.h"

int main() {

    // ***** PROGRAM DEFINITION *****
    // Our first, simple program.
    auto input = mk_array("input");
    auto output = mk_array("output");

    gern::annot::add_1 add_1;

    Composable program = {
        add_1(input, output),
    };
}
