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
template<typename T1, typename T2, typename T3>
static void open_blas_mm(T1 &a, T2 &b, T3 &c) {
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

std::function<void(impl::MatrixViewPackOnQuery &a, impl::MatrixViewPackOnQuery &b, impl::MatrixViewBasic &c)> get_inner_kernel_gern_packed() {
    constexpr int64_t outer_tile = 256;

    const std::string templated_matrix = "MatrixTemplated<" + std::to_string(outer_tile) + ", " + std::to_string(outer_tile) + ">";

    auto A_in = AbstractDataTypePtr(new const annot::Matrix(templated_matrix, false, "A"));
    auto B_in = AbstractDataTypePtr(new const annot::Matrix(templated_matrix, false, "B"));
    auto C_in = AbstractDataTypePtr(new const annot::Matrix("MatrixViewBasic", false, "C"));

    annot::MatrixMultiply matrix_multiply("inner_kernel", A_in, B_in, C_in);

    Variable ti_in("ti_in");
    Variable tj_in("tj_in");
    Variable kdim_in("kdim_in");

    Composable program_in{
        Tile(C_in["row"], ti_in.bind(8))(
            Tile(C_in["col"], tj_in.bind(16))(
                matrix_multiply(A_in, B_in, C_in, kdim_in)))};

    auto options_in = benchmark::cpuRunner();
    options_in.filename = "benchmark_in.cpp";

    FunctionPtr func_in_pre{program_in, options_in, std::vector<Parameter>{Parameter{A_in}, Parameter{B_in}, Parameter{C_in}, Parameter{kdim_in}}};
    auto func_in = &func_in_pre[{{"ti_in", ti_in.bind(8)}, {"tj_in", tj_in.bind(16)}}];

    auto A_out = AbstractDataTypePtr(new const annot::Matrix("MatrixViewPackOnQuery", true, "A_out"));
    auto B_out = AbstractDataTypePtr(new const annot::Matrix("MatrixViewPackOnQuery", true, "B_out"));
    auto C_out = AbstractDataTypePtr(new const annot::Matrix("MatrixViewBasic", false, "C"));

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

std::function<void(impl::A3 &a, impl::B3 &b, impl::MatrixViewBasic &c)> get_blis() {
    constexpr int64_t outer_tile = 192 * 2;

    const std::string templated_matrix_a = "A2<" + std::to_string(outer_tile) + ", " + std::to_string(outer_tile) + ">";
    const std::string templated_matrix_b = "B2<" + std::to_string(outer_tile) + ", " + std::to_string(outer_tile) + ">";

    auto A_in = AbstractDataTypePtr(new const annot::Matrix(templated_matrix_a, false, "A"));
    auto B_in = AbstractDataTypePtr(new const annot::Matrix(templated_matrix_b, false, "B"));
    auto C = AbstractDataTypePtr(new const annot::Matrix("MatrixViewBasic", false, "C"));

    annot::MatrixMultiply matrix_multiply("blis_kernel", A_in, B_in, C);

    Variable ti_in("ti_in");
    Variable tj_in("tj_in");
    Variable kdim_in("kdim_in");

    Composable program_in{
        Tile(C["col"], tj_in.bind(impl::internal::kernel_cols))(
            Tile(C["row"], ti_in.bind(impl::internal::kernel_rows))(
                matrix_multiply(A_in, B_in, C, kdim_in)))};

    auto options_in = benchmark::cpuRunner();
    options_in.filename = "benchmark_in.cpp";

    FunctionPtr func_in_pre{program_in, options_in, std::vector<Parameter>{Parameter{A_in}, Parameter{B_in}, Parameter{C}, Parameter{kdim_in}}};
    auto func_in = &func_in_pre[{{"ti_in", ti_in.bind(impl::internal::kernel_rows)}, {"tj_in", tj_in.bind(impl::internal::kernel_cols)}}];

    auto A_out = AbstractDataTypePtr(new const annot::Matrix("A3", true, "A_out"));
    auto B_out = AbstractDataTypePtr(new const annot::Matrix("B3", true, "B_out"));

    Variable ti_out("ti_out");
    Variable tj_out("tj_out");
    Variable tk_out("tk_out");
    Variable kdim_out("kdim_out");

    Composable program_out{
        Tile(C["col"], tj_out.bind(outer_tile))(
            Reduce(kdim_out, tk_out.bind(outer_tile))(
                Tile(C["row"], ti_out.bind(outer_tile))(
                    (*func_in)(A_out, B_out, C, kdim_out))))};

    Runner run(program_out);
    run.compile(benchmark::cpuRunner());

    return [run,
            A_name = A_out.getName(),
            B_name = B_out.getName(),
            C_name = C.getName(),
            kdim_out_name = kdim_out.getName()](impl::A3 &a, impl::B3 &b, impl::MatrixViewBasic &c) mutable {
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
static double run_function(std::function<void(T1 &, T2 &, T3 &)> func, std::string func_name, bool warmup, int64_t samples, int64_t iterations,
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

    return time;
}
int main() {
    openblas_set_num_threads(1);
    // std::cout << openblas_get_num_threads() << std::endl;


    std::vector<std::pair<int64_t, double>> data;

    for (int64_t cnt = 1; cnt <= 10; ++cnt) {

        int64_t n = cnt * 256, m = cnt * 256, k = cnt * 256;

        std::cout << "Running with dimensions: " << n << ", " << m << ", " << k << std::endl;
        
        impl::MatrixViewPackOnQuery a_pack = impl::MatrixViewPackOnQuery::allocate(0, 0, n, k);
        impl::MatrixViewPackOnQuery b_pack = impl::MatrixViewPackOnQuery::allocate(0, 0, k, m);
        impl::MatrixViewBasic res = impl::MatrixViewBasic::allocate(0, 0, n, m);
        impl::MatrixViewBasic c = impl::MatrixViewBasic::allocate(0, 0, n, m);


        random_fill(a_pack);
        random_fill(b_pack);

        // run_function<impl::MatrixViewPackOnQuery, impl::MatrixViewPackOnQuery, impl::MatrixViewBasic>(
        //     open_blas_mm<impl::MatrixViewPackOnQuery, impl::MatrixViewPackOnQuery, impl::MatrixViewBasic>, "open_blas", true, 1, 1, a_pack, b_pack, res, false, c);

        double time = run_function<impl::MatrixViewPackOnQuery, impl::MatrixViewPackOnQuery, impl::MatrixViewBasic>(
            get_inner_kernel_gern_packed(), "inner_kernel_gern_packed", true, 1, 1, a_pack, b_pack, c, false, res);

        data.push_back({n, time});

#ifdef BLIS_KERNEL
        impl::A3 a3 = impl::A3::allocate(0, 0, n, k);
        impl::B3 b3 = impl::B3::allocate(0, 0, k, m);

        impl::copy_data(a_pack, a3);
        impl::copy_data(b_pack, b3);

        run_function<impl::A3, impl::B3, impl::MatrixViewBasic>(
            get_blis(), "blis", false, 1, 1, a3, b3, c, true, res);

        a3.destroy();
        b3.destroy();
#endif

        c.destroy();
        a_pack.destroy();
        b_pack.destroy();

        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Data for roofline curve" << std::endl;
    for (auto &[n, sec] : data) {
        std::cout << n/12 << std::endl << 2 * n*n*n / sec << std::endl;
    }
    std::cout << std::endl;
}