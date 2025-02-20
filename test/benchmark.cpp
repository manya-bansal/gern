#include "my_benchmark.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"

#include "cblas.h"
#include "openblas_config.h"



using namespace gern;

static void basic_mm(impl::MatrixCPU &a, impl::MatrixCPU &b, impl::MatrixCPU &c) {
    int64_t N = c.row, M = c.col, K = a.col;
    
    for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < M; ++j)
    for (int64_t k = 0; k < K; ++k)
        c.data[i*c.lda + j] += a.data[i*a.lda + k] * b.data[k*b.lda + j];
}

static void open_blas_mm(impl::MatrixCPU &a, impl::MatrixCPU &b, impl::MatrixCPU &c) {
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
        c.data, M
    );
}

static std::function<void(impl::MatrixCPU &, impl::MatrixCPU &, impl::MatrixCPU &)> get_gern_basic_tiling() {
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixCPU("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixCPU("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixCPU("C"));

    annot::MatrixMultiplyCPU matrix_multiply;
    Variable ti("ti");
    Variable tj("tj");
    Variable k("k");
    Variable k_dim("k_dim");

    Composable program = {
        (Tile(C_DS["row"], ti))(
            Tile(C_DS["col"], tj)(
                (Reduce(k_dim, k))(
                    matrix_multiply(A_DS, B_DS, C_DS, k_dim))))
    };

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    return [
        run,
        A_name = A_DS.getName(),
        B_name = B_DS.getName(),
        C_name = C_DS.getName(),
        ti_name = ti.getName(),
        tj_name = tj.getName(),
        k_name = k.getName(),
        k_dim_name = k_dim.getName()
    ] (impl::MatrixCPU &a, impl::MatrixCPU &b, impl::MatrixCPU &c) mutable {
        int64_t K = a.col;
        int64_t tile_size = 128;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {ti_name, &tile_size},
            {tj_name, &tile_size},
            {k_name, &tile_size},
            {k_dim_name, &K},
        });
    };
}


static std::function<void(impl::MatrixCPU &, impl::MatrixCPU &, impl::MatrixCPU &)> get_gern_double_tiling() {
    
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixCPU("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixCPU("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixCPU("C"));

    annot::MatrixMultiplyCPU matrix_multiply;
    Variable ti("ti");
    Variable ti_2("ti_2");
    Variable tj("tj");
    Variable tj_2("tj_2");
    Variable k("k");
    Variable k_2("k_2");
    Variable k_dim("k_dim");

    Composable program = {
        (Tile(C_DS["row"], ti))(
            Tile(C_DS["col"], tj)(
                (Reduce(k_dim, k))(
                    (Tile(C_DS["row"], ti_2))(
                        Tile(C_DS["col"], tj_2)(
                            Reduce(k_dim, k_2)(
                                matrix_multiply(A_DS, B_DS, C_DS, k_dim)))))))};

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    return [
        run,
        A_name = A_DS.getName(),
        B_name = B_DS.getName(),
        C_name = C_DS.getName(),
        ti_name = ti.getName(),
        ti_2_name = ti_2.getName(),
        tj_name = tj.getName(),
        tj_2_name = tj_2.getName(),
        k_name = k.getName(),
        k_2_name = k_2.getName(),
        k_dim_name = k_dim.getName()
    ] (impl::MatrixCPU &a, impl::MatrixCPU &b, impl::MatrixCPU &c) mutable {
        int64_t K = a.col;

        int64_t tile_size = 128;
        int64_t tile_size_2 = 16;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {ti_name, &tile_size},
            {ti_2_name, &tile_size_2},
            {tj_name, &tile_size},
            {tj_2_name, &tile_size_2},
            {k_name, &tile_size},
            {k_2_name, &tile_size_2},
            {k_dim_name, &K},
        });
    };
}

static std::function<void(impl::MatrixCPUPack &, impl::MatrixCPUPack &, impl::MatrixCPU &)> get_gern_packed() {
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixCPUPack("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixCPUPack("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixCPU("C"));

    annot::MatrixMultiplyCPUPacked matrix_multiply;
    Variable ti("ti");
    Variable tj("tj");
    Variable k("k");
    Variable k_dim("k_dim");

    Composable program = {
        (Tile(C_DS["row"], ti))(
            (Reduce(k_dim, k))(
            Tile(C_DS["col"], tj)(
                    matrix_multiply(A_DS, B_DS, C_DS, k_dim))))
    };

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));

    return [
        run,
        A_name = A_DS.getName(),
        B_name = B_DS.getName(),
        C_name = C_DS.getName(),
        ti_name = ti.getName(),
        tj_name = tj.getName(),
        k_name = k.getName(),
        k_dim_name = k_dim.getName()
    ] (impl::MatrixCPUPack &a, impl::MatrixCPUPack &b, impl::MatrixCPU &c) mutable {
        int64_t K = a.col;
        int64_t tile_size = 128;

        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {ti_name, &tile_size},
            {tj_name, &tile_size},
            {k_name, &tile_size},
            {k_dim_name, &K},
        });
    };
}


static std::function<void(impl::MatrixCPU &, impl::MatrixCPU &, impl::MatrixCPU &)> get_gern_complex() {
    auto A_DS = AbstractDataTypePtr(new const annot::MatrixCPU("A"));
    auto B_DS = AbstractDataTypePtr(new const annot::MatrixCPU("B"));
    auto C_DS = AbstractDataTypePtr(new const annot::MatrixCPU("C"));

    annot::MatrixMultiplyCPU matrix_multiply;
    Variable ti("ti");
    Variable ti_g("ti_g");
    Variable ti_2("ti_2");
    Variable tj("tj");
    Variable tj_g("tj_g");
    Variable tj_2("tj_2");
    Variable tk("tk");
    Variable inner_k("inner_k");

    Variable k_2("k_2");
    Variable k_dim("k_dim");
    // Just bind all, dont want to deal with args rn.
    Composable tile_mm_registers = {
        (Tile(C_DS["row"], ti))(
            Tile(C_DS["col"], tj)(
                (Tile(C_DS["row"], ti_2))(
                    Tile(C_DS["col"], tj_2)(
                        Reduce(tk, inner_k)(
                            matrix_multiply(A_DS, B_DS, C_DS, tk))))))};

    Runner::Options options = test::cpuRunner("matrix");
    options.filename = "tile_mm_registers.cpp";
    FunctionPtr tile_mm_registers_ptr(tile_mm_registers, options);

    Composable load_into_shared = {
        (Reduce(k_dim, tk))(
            tile_mm_registers_ptr(A_DS, B_DS, C_DS,
                                  inner_k, ti, ti_2, tj, tj_2, k_dim))};

    options.filename = "load_into_shared.cpp";
    FunctionPtr load_into_shared_ptr(load_into_shared, options);

    Composable program = {
        (Tile(C_DS["row"], ti_g))(
            Tile(C_DS["col"], tj_g)(
                load_into_shared_ptr(A_DS, B_DS, C_DS,
                                     inner_k, k_dim, ti, ti_2, tj, tj_2, tk))),
    };

    Runner run(program);
    run.compile(test::cpuRunner(std::vector<std::string>{"matrix"}));


    return [
        run,
        A_name = A_DS.getName(),
        B_name = B_DS.getName(),
        C_name = C_DS.getName(),

        ti_name = ti.getName(),
        ti_g_name = ti_g.getName(),
        ti_2_name = ti_2.getName(),

        tj_name = tj.getName(),
        tj_g_name = tj_g.getName(),
        tj_2_name = tj_2.getName(),

        tk_name = tk.getName(),
        inner_k_name = inner_k.getName(),
        k_dim_name = k_dim.getName()
    ] (impl::MatrixCPU &a, impl::MatrixCPU &b, impl::MatrixCPU &c) mutable {
        int64_t N = a.row, M = b.col, K = a.col;

        int64_t ti_val = 128;
        int64_t ti_2_val = 16;
    
        int64_t tj_val = 128;
        int64_t tj_2_val = 16;
    
    
        int64_t tk_val = 128;
        int64_t inner_k_val = 16;
        
        run.evaluate({
            {A_name, &a},
            {B_name, &b},
            {C_name, &c},
            {ti_name, &ti_val},
            {ti_g_name, &N},
            {ti_2_name, &ti_2_val},
            {tj_name, &tj_val},
            {tj_g_name, &M},
            {tj_2_name, &tj_2_val},
            {tk_name, &tk_val},
            {inner_k_name, &inner_k_val},
            {k_dim_name, &K},
        });
    };
}


void pack(impl::MatrixCPU a, float *res) {
    for (int64_t i = 0; i < a.row; ++i) {
        std::copy(a.data + i*a.lda, a.data + i*a.lda + a.col, res);
        res += a.col;
    }
}

void tiled_packed_mult(impl::MatrixCPU &a, impl::MatrixCPU &b, impl::MatrixCPU &c) {
    constexpr int64_t tile_size = 128;

    assert(a.row % tile_size == 0 && a.col % tile_size == 0 && c.col % tile_size == 0);

    float a_packed[tile_size][tile_size];
    float b_packed[tile_size][tile_size];

    for (int64_t i_out = 0; i_out < a.row; i_out += tile_size)
    for (int64_t k_out = 0; k_out < a.col; k_out += tile_size)
    for (int64_t j_out = 0; j_out < b.col; j_out += tile_size) {

        pack(a.query(i_out, k_out, tile_size, tile_size), a_packed[0]);
        pack(b.query(k_out, j_out, tile_size, tile_size), b_packed[0]);

        for (int64_t i = 0; i < tile_size; ++i)
        for (int64_t k = 0; k < tile_size; ++k)
        for (int64_t j = 0; j < tile_size; ++j)
            c.data[(i_out + i) * c.lda + j_out + j] += a_packed[i][k] * b_packed[k][j];
    } 
}


template <typename InputType, typename ResType>
static void run_function(std::function<void(InputType &, InputType &, ResType &)> func, std::string func_name, bool warmup, int64_t samples, int64_t iterations,
InputType &a, InputType &b, ResType &c, bool check_res, ResType res) {
 
    if (warmup) {
        c.vvals(0);
        func(a, b, c);
    }

    double time = my_benchmark::benchmark(samples, iterations, [&] {
        c.vvals(0);
        func(a, b, c);
    });

    std::cout << "Function " << func_name << " ran in " << time << " seconds" << std::endl;

    if (check_res) {
        bool correct = true;

        for (int64_t i = 0; i < c.row; ++i) {
            for (int64_t j = 0; j < c.col; ++j) {
                if (abs(c.data[i*c.lda + j] - res.data[i*res.lda + j]) > 0.01) {
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


    int64_t n = 2 * 1024, m = 2 * 1024, k = 2 * 1024;
    
    impl::MatrixCPU a{n, k, k}, b{k, m, m}, c{n, m, m}, res{n, m, m};
    impl::MatrixCPUPack a_packed{n, k, k}, b_packed{k, m, m};

    
    a.random_fill();
    b.random_fill();
    std::copy(a.data, a.data + n*k, a_packed.data);
    std::copy(b.data, b.data + k*m, b_packed.data);

    // run_function(basic_mm, "basic_mm", false, 1, 1, a, b, res, false, c);
    run_function<impl::MatrixCPU, impl::MatrixCPU>(open_blas_mm, "open_blas_mm", true, 1, 1, a, b, res, false, c);
    run_function<impl::MatrixCPU, impl::MatrixCPU>(tiled_packed_mult, "tiled_packed_mult", true, 1, 1, a, b, c, true, res);
    // run_function<impl::MatrixCPU, impl::MatrixCPU>(get_gern_basic_tiling(), "gern_basic_tiling", true, 1, 1, a, b, c, true, res);
    run_function(get_gern_double_tiling(), "gern_double_tiling", true, 1, 1, a, b, c, true, res);
    run_function<impl::MatrixCPUPack, impl::MatrixCPU>(get_gern_packed(), "gern_packed", true, 1, 1, a_packed, b_packed, c, true, res);
    // run_function(get_gern_complex(), "gern_complex", true, 1, 1, a, b, c, true, res);
}