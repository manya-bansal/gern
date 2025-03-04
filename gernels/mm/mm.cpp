
#include "../current_path.h"
#include "compose/runner.h"
#include "gern_annot/functions.h"
#include "wrappers/adt.h"
#include <assert.h>
#include <iostream>

using namespace gern;

int main() {
    auto input = AbstractDataTypePtr(new const annot::ArrayGPU("input"));
    auto output = AbstractDataTypePtr(new const annot::FloatPtr("output"));
}