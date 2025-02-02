#include "annot/adt.h"
#include "annot/functions.h"
#include "compose/composable.h"
#include "compose/runner.h"
#include "gpu-matrix-const.h"

int main() {
    constexpr int64_t h = 8;
    constexpr int64_t w = 8;
    constexpr int64_t stride = 1;

    using MatrixType = impl::MatrixGPU<h, w, h, stride>;

    auto a = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, stride>("input", false));
    auto b = AbstractDataTypePtr(new const annot::MatrixGPU<h, w, stride>("output", false));

    annot::ExpMatrix exp_matrix{a, b};

    Variable row("row");
    Variable col("col");

    Composable program = {
        Global(Tile(b["row"], row.bindToInt64(2))(
            Tile(b["col"], col.bindToInt64(4))(
                exp_matrix(a, b))))};

    Runner run(program);
    Runner::Options options;

    options.include = "-I /home/manya/gern/apps/"
                      " -I /home/manya/gern/test/";
    options.arch = "89";

    run.compile(options);

    MatrixType in;
    in.ascending();
    MatrixType out;
    out.vvals(0.0f);

    run.evaluate({
        {a.getName(), &in},
        {b.getName(), &out},
    });

    auto cpu_result = out.get();
    auto a_result = in.get();
    std::cout << cpu_result << std::endl;
    std::cout << a_result << std::endl;
}