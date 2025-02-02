#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"

int main() {
    constexpr int64_t h = 16;
    constexpr int64_t w = 16;
    constexpr int64_t stride = 1;

    auto a = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, stride>("input", false));
    auto b = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, stride>("output", false));

    annot::ExpMatrix exp_matrix{a, b};

    Variable row("row");
    Variable col("col");

    Composable program = {
        Global(Tile(b["row"], row.bindToInt64(8))(
            Tile(b["col"], col.bindToInt64(8))(
                exp_matrix(a, b))))};

    Runner run(program);
    Runner::Options options;

    options.include = "-I /home/manya/gern/apps/"
                      " -I /home/manya/gern/test/";
    options.arch = "89";

    run.compile(options);
}