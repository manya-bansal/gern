#include "library/annot.h"
#include <compose/composable.h>
#include <compose/runner.h>

#include <iostream>

using namespace gern;

int main() {
    Hello hi;

    Composable program(hi());
    Runner run(program);

    run.compile(Runner::Options());
    // run.evaluate({});

    return 0;
}