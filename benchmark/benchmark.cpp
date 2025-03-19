#include "benchmark-utils.h"
#include "lib/annot/cpu-matrix.h"
#include "lib/impl/cpu-matrix.h"

#include "cblas.h"
#include "openblas_config.h"

using namespace gern;

template<typename T>
static void basic_mm(T &a, T &b, T &c) {
    int64_t N = c.row, M = c.col, K = a.col;

    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < M; ++j)
            for (int64_t k = 0; k < K; ++k)
                c.get(i, j) += a.get(i, k) * b.get(k, j);
}
template<typename T>
static void open_blas_mm(T &a, T &b, T &c) {
    int64_t N = c.row, M = c.col, K = a.col;

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, M, K,
        1,
        a.data, K,
        b.data, M,
        1,
        c.data, M);
}
static std::function<void(impl::MatrixViewBasic &, impl::MatrixViewBasic &, impl::MatrixViewBasic &)> get_gern_double_tiling() {

    auto A_DS = AbstractDataTypePtr(new const annot::MatrixViewBasic("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixViewBasic("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixViewBasic("C"));

    annot::MatrixMultiplyGeneric<annot::MatrixViewBasic, annot::MatrixViewBasic, annot::MatrixViewBasic> matrix_multiply;
    Variable ti("ti");
    Variable ti_2("ti_2");
    Variable tj("tj");
    Variable tj_2("tj_2");
    Variable k("k");
    Variable k_2("k_2");
    Variable k_dim("k_dim");

    constexpr int64_t tile_size = 128;
    constexpr int64_t tile_size_2 = 16;

    Composable program = {
        Reduce(k_dim, k.bind(tile_size))(
            Tile(C_DS["row"], ti.bind(tile_size))(
                Tile(C_DS["col"], tj.bind(tile_size))(
                    Reduce(k_dim, k_2.bind(tile_size_2))(
                        Tile(C_DS["row"], ti_2.bind(tile_size_2))(
                            Tile(C_DS["col"], tj_2.bind(tile_size_2))(
                                matrix_multiply(A_DS, B_DS, C_DS, k_dim)))))))};

    Runner run(program);
    run.compile(benchmark::cpuRunner());

    return [run,
            A_name = A_DS.getName(),
            B_name = B_DS.getName(),
            C_name = C_DS.getName(),
            k_dim_name = k_dim.getName()](impl::MatrixViewBasic &a, impl::MatrixViewBasic &b, impl::MatrixViewBasic &c) mutable {
        int64_t K = a.col;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {k_dim_name, &K},
        });
    };
}
static std::function<void(impl::MatrixViewPackOnQuery &, impl::MatrixViewPackOnQuery &, impl::MatrixViewBasic &)> get_gern_packed() {
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixViewPackOnQuery("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixViewPackOnQuery("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixViewBasic("C"));

    constexpr int64_t tile_size = 128;

    annot::MatrixMultiplyTemplated<tile_size, tile_size, tile_size, annot::MatrixViewBasic> matrix_multiply;

    Variable ti("ti");
    Variable tj("tj");
    Variable k("k");
    Variable k_dim("k_dim");

    Composable program = {
        Tile(C_DS["row"], ti.bind(tile_size))(
            Reduce(k_dim, k.bind(tile_size))(
                Tile(C_DS["col"], tj.bind(tile_size))(
                    matrix_multiply(A_DS, B_DS, C_DS, k_dim))))};

    Runner run(program);
    run.compile(benchmark::cpuRunner());

    return [run,
            A_name = A_DS.getName(),
            B_name = B_DS.getName(),
            C_name = C_DS.getName(),
            k_dim_name = k_dim.getName()](impl::MatrixViewPackOnQuery &a, impl::MatrixViewPackOnQuery &b, impl::MatrixViewBasic &c) mutable {
        int64_t K = a.col;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {k_dim_name, &K},
        });
    };
}
void pack(impl::MatrixViewBasic a, float *res) {
    for (int64_t i = 0; i < a.row; ++i) {
        std::copy(&a.get(i, 0), &a.get(i, a.col), res);
        res += a.col;
    }
}
void tiled_packed_mult(impl::MatrixViewBasic &a, impl::MatrixViewBasic &b, impl::MatrixViewBasic &c) {
    constexpr int64_t tile_size = 128;

    assert(a.row % tile_size == 0 && a.col % tile_size == 0 && c.col % tile_size == 0);

    float a_packed[tile_size][tile_size];
    float b_packed[tile_size][tile_size];

    for (int64_t i_out = 0; i_out < a.row; i_out += tile_size)
        for (int64_t k_out = 0; k_out < a.col; k_out += tile_size) {
            pack(a.query(i_out, k_out, tile_size, tile_size), a_packed[0]);

            for (int64_t j_out = 0; j_out < b.col; j_out += tile_size) {
                pack(b.query(k_out, j_out, tile_size, tile_size), b_packed[0]);

                for (int64_t i = 0; i < tile_size; ++i)
                    for (int64_t k = 0; k < tile_size; ++k)
                        for (int64_t j = 0; j < tile_size; ++j)
                            c.get(i_out + i, j_out + j) += a_packed[i][k] * b_packed[k][j];
            }
        }
}
void inner_kernel_handwritten(impl::MatrixViewBasic &a, impl::MatrixViewBasic &b, impl::MatrixViewBasic &c) {
    int64_t N = c.row, M = c.col, K = a.col;

    constexpr int64_t tile_size = 128;

    for (int64_t i_out = 0; i_out < N; i_out += tile_size)
        for (int64_t k = 0; k < K; k += tile_size)
            for (int64_t j_out = 0; j_out < M; j_out += tile_size)
                for (int64_t i = i_out; i < i_out + tile_size; i += 8)
                    for (int64_t j = j_out; j < j_out + tile_size; j += 16) {

                        impl::MatrixViewBasic a_tmp = a.query(i, k, 8, tile_size);
                        impl::MatrixViewBasic b_tmp = b.query(k, j, tile_size, 16);
                        impl::MatrixViewBasic c_tmp = c.query(i, j, 8, 16);

                        impl::inner_kernel(a_tmp, b_tmp, c_tmp, 0);
                    }
}
std::function<void(impl::MatrixViewBasic &a, impl::MatrixViewBasic &b, impl::MatrixViewBasic &c)> get_inner_kernel_gern() {

    auto A = AbstractDataTypePtr(new const annot::MatrixViewBasic("A"));
    auto B = AbstractDataTypePtr(new const annot::MatrixViewBasic("B"));
    auto C = AbstractDataTypePtr(new const annot::MatrixViewBasic("C"));

    annot::MatrixMultiplyKernel matrix_multiply;

    Variable ti_in("ti_in");
    Variable tj_in("tj_in");
    Variable kdim_in("kdim_in");

    Variable ti_out("ti_out");
    Variable tj_out("tj_out");
    Variable tk_out("tk_out");
    Variable kdim("kdim");

    int64_t tile_size = 256;

    Composable program{
        Tile(C["row"], ti_out.bind(tile_size))(
            Reduce(kdim, tk_out.bind(tile_size))(
                Tile(C["col"], tj_out.bind(tile_size))(
                    Tile(C["row"], ti_in.bind(8))(
                        Tile(C["col"], tj_in.bind(16))(
                            matrix_multiply(A, B, C, kdim))))))};

    Runner run(program);
    run.compile(benchmark::cpuRunner());

    return [run,
            A_name = A.getName(),
            B_name = B.getName(),
            C_name = C.getName(),
            kdim_name = kdim.getName()](impl::MatrixViewBasic &a, impl::MatrixViewBasic &b, impl::MatrixViewBasic &c) mutable {
        int64_t K = a.col;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {kdim_name, &K},
        });
    };
}
std::function<void(impl::MatrixViewPackOnQuery &a, impl::MatrixViewPackOnQuery &b, impl::MatrixViewBasic &c)> get_inner_kernel_gern_packed() {
    constexpr int64_t outer_tile = 256;

    auto A_in = AbstractDataTypePtr(new const annot::MatrixTemplated<outer_tile, outer_tile>("A"));
    auto B_in = AbstractDataTypePtr(new const annot::MatrixTemplated<outer_tile, outer_tile>("B"));
    auto C_in = AbstractDataTypePtr(new const annot::MatrixViewBasic("C"));

    annot::MatrixMultiplyKernel matrix_multiply;

    Variable ti_in("ti_in");
    Variable tj_in("tj_in");
    Variable kdim_in("kdim_in");

    Composable program_in{
        Tile(C_in["row"], ti_in.bind(8))(
            Tile(C_in["col"], tj_in.bind(16))(
                matrix_multiply(A_in, B_in, C_in, kdim_in)))};

    auto options_in = benchmark::cpuRunner();
    options_in.filename = "benchmark_in.cpp";

    FunctionPtr func_in_pre{program_in, options_in};
    auto func_in = &func_in_pre[{{"ti_in", ti_in.bind(8)}, {"tj_in", tj_in.bind(16)}}];

    auto A_out = AbstractDataTypePtr(new const annot::MatrixViewPackOnQuery("A_out"));
    auto B_out = AbstractDataTypePtr(new const annot::MatrixViewPackOnQuery("B_out"));
    auto C_out = AbstractDataTypePtr(new const annot::MatrixViewBasic("C_out"));

    Variable ti_out("ti_out");
    Variable tj_out("tj_out");
    Variable tk_out("tk_out");
    Variable kdim_out("kdim_out");

    Composable program_out{
        Tile(C_out["row"], ti_out.bind(outer_tile))(
            Reduce(kdim_out, tk_out.bind(outer_tile))(
                Tile(C_out["col"], tj_out.bind(outer_tile))(
                    (*func_in)(A_out, B_out, C_out, kdim_out))))};

    Runner run(program_out);
    run.compile(benchmark::cpuRunner());

    return [run,
            A_name = A_out.getName(),
            B_name = B_out.getName(),
            C_name = C_out.getName(),
            kdim_out_name = kdim_out.getName()](impl::MatrixViewPackOnQuery &a, impl::MatrixViewPackOnQuery &b, impl::MatrixViewBasic &c) mutable {
        int64_t K = a.col;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {kdim_out_name, &K},
        });
    };
}
template<typename T1, typename T2, typename T3, typename T4>
static void run_function(std::function<void(T1 &, T2 &, T3 &)> func, std::string func_name, bool warmup, int64_t samples, int64_t iterations,
                         T1 &a, T2 &b, T3 &c, bool check_res, T4 &res) {

    if (warmup) {
        impl::fill_with(c, 0);
        func(a, b, c);
    }

    double time = benchmark::benchmark(samples, iterations, [&] {
        impl::fill_with(c, 0);
        func(a, b, c);
    });

    std::cout << "Function " << func_name << " ran in " << time << " seconds" << std::endl;

    if (check_res) {
        bool correct = true;

        for (int64_t i = 0; i < c.row; ++i) {
            for (int64_t j = 0; j < c.col; ++j) {
                if (abs(c.get(i, j) - res.get(i, j)) > 0.01) {
                    correct = false;
                }
            }
        }
        if (!correct) {
            std::cout << "INCORRECT FUNCTION: " << func_name << std::endl;
        }
    }
}
int main() {
    openblas_set_num_threads(1);

    int64_t n = 6 * 1024, m = 6 * 1024, k = 6 * 1024;

    impl::MatrixViewBasic a = impl::MatrixViewBasic::allocate(0, 0, n, k);
    impl::MatrixViewBasic b = impl::MatrixViewBasic::allocate(0, 0, k, m);
    impl::MatrixViewBasic c = impl::MatrixViewBasic::allocate(0, 0, n, m);

    impl::MatrixViewPackOnQuery a_pack = impl::MatrixViewPackOnQuery::allocate(0, 0, n, k);
    impl::MatrixViewPackOnQuery b_pack = impl::MatrixViewPackOnQuery::allocate(0, 0, k, m);

    impl::MatrixViewBasic res = impl::MatrixViewBasic::allocate(0, 0, n, m);

    random_fill(a);
    std::copy(a.data, a.data + n * k, a_pack.data);
    random_fill(b);
    std::copy(b.data, b.data + k * m, b_pack.data);

    // run_function(basic_mm<impl::MatrixViewBasic>, "basic_mm", false, 1, 1, a, b, res, false, c);

    run_function<impl::MatrixViewBasic, impl::MatrixViewBasic, impl::MatrixViewBasic>(
        open_blas_mm<impl::MatrixViewBasic>, "open_blas", false, 1, 1, a, b, res, false, c);
    // run_function<impl::MatrixViewBasic, impl::MatrixViewBasic, impl::MatrixViewBasic>(
    //     tiled_packed_mult, "hand_packed", false, 1, 1, a, b, c, true, res);

    // // run_function<impl::MatrixViewBasic, impl::MatrixViewBasic, impl::MatrixViewBasic>(
    // //     get_gern_double_tiling(), "gern_double_tiled", false, 1, 1, a, b, c, true, res);

    // run_function<impl::MatrixViewPackOnQuery, impl::MatrixViewPackOnQuery, impl::MatrixViewBasic>(
    //     get_gern_packed(), "gern_packed", false, 1, 1, a_pack, b_pack, c, true, res);

    // run_function<impl::MatrixViewBasic, impl::MatrixViewBasic, impl::MatrixViewBasic>(
    //     inner_kernel_handwritten, "inner_kernel_handwritten", false, 1, 1, a, b, c, true, res
    // );

    // run_function<impl::MatrixViewBasic, impl::MatrixViewBasic, impl::MatrixViewBasic>(
    //     get_inner_kernel_gern(), "inner_kernel_gern", false, 1, 1, a, b, c, true, res
    // );

    run_function<impl::MatrixViewPackOnQuery, impl::MatrixViewPackOnQuery, impl::MatrixViewBasic>(
        get_inner_kernel_gern_packed(), "inner_kernel_gern_packed", false, 1, 1, a_pack, b_pack, c, true, res);
}