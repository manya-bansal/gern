import torch
import torch.fx
import torch.utils.benchmark as benchmark
import operator
from generate_torch_compile_pipeline import gen, FnInterface
import math
from gern_py import *
import time

torch_to_gern = {
    torch.t: FnInterface(MatrixTranspose),
    operator.matmul: FnInterface(MatrixMultiply, {"shared_len": lambda args: args[0].meta["example_value"].size(dim=1)}),
    operator.truediv: FnInterface(MatrixDivn),
    torch.nn.functional.softmax: FnInterface(MatrixSoftmax)
}

class M(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.softmax((q @ torch.t(k)) / math.sqrt(q.size(dim=1))) @ v

def benchmark_runtimes(rows, cols, tiling):
    q = torch.randn((rows, cols))
    k = torch.randn((rows, cols))
    v = torch.randn((rows, cols))

    results = []

    for row_tiling in tiling:
        torch.compiler.reset()
        opt_M = gen(M, torch_to_gern, tile_rows=row_tiling)

        optimized = benchmark.Timer(
            setup='opt_M(q, k, v)',
            stmt='opt_M(q, k, v)',
            globals={'opt_M': opt_M, 'q': q, 'k': k, 'v': v},
            label=f"flash attention {rows} x {cols}",
            description=f"{rows} x {cols}",
            sub_label=f"gern, row tiling {row_tiling}"
        )

        results.append(optimized.blocked_autorange(min_run_time=2))

    m = M()
    
    unoptimized = benchmark.Timer(
        stmt='m(q, k, v)',
        globals={'m': m, 'q': q, 'k': k, 'v': v},
        label=f"flash attention {rows} x {cols}",
        description=f"{rows} x {cols}",
        sub_label="unoptimized"
    )

    torch.compiler.reset()
    default_tc_m = torch.compile(M())

    default_compiled = benchmark.Timer(
        setup='default_tc_m(q, k, v)',
        stmt='default_tc_m(q, k, v)',
        globals={'default_tc_m': default_tc_m, 'q': q, 'k': k, 'v': v},
        label=f"flash attention {rows} x {cols}",
        description=f"{rows} x {cols}",
        sub_label="default torch.compile"
    )

    q_4d = q[None, None, :, :]
    k_4d = k[None, None, :, :]
    v_4d = v[None, None, :, :]

    flash_attention = benchmark.Timer(
        stmt="""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        scaled_dot_product_attention(q, k, v)
    """,
        setup="""
    from torch.nn.functional import scaled_dot_product_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    """,
        globals={'q': q_4d, 'k': k_4d, 'v': v_4d},
        label=f"flash attention {rows} x {cols}",
        description=f"{rows} x {cols}",
        sub_label="pytorch flash attention"
    )

    
    results.append(unoptimized.blocked_autorange(min_run_time=2))
    results.append(default_compiled.blocked_autorange(min_run_time=2))
    results.append(flash_attention.blocked_autorange(min_run_time=2))

    compare = benchmark.Compare(results)
    compare.print()

    return results

def benchmark_module(M, label, row_tilings, *args):
    results = []
        
    for row_tiling in row_tilings:
        torch.compiler.reset()
        opt_M = gen(M, torch_to_gern, tile_rows=row_tiling)
    
        gern = benchmark.Timer(
                setup=f'opt_M({','.join(arg_name for arg_name, _ in args)})',
                stmt=f'opt_M({','.join(arg_name for arg_name, _ in args)})',
                globals={'opt_M': opt_M} | { arg_name : arg_val for arg_name, arg_val in args },
                label=label,
                description=label,
                sub_label=f"gern, row tiling {row_tiling}"
            )

        results.append(gern.blocked_autorange(min_run_time=2))
    
    m = M()
    
    unoptimized = benchmark.Timer(
        stmt=f'm({','.join(arg_name for arg_name, _ in args)})',
        globals={'m': m} | { arg_name : arg_val for arg_name, arg_val in args },
        label=label,
        description=label,
        sub_label="unoptimized"
    )

    torch.compiler.reset()
    default_tc_m = torch.compile(M())

    default_compiled = benchmark.Timer(
        setup=f'default_tc_m({','.join(arg_name for arg_name, _ in args)})',
        stmt=f'default_tc_m({','.join(arg_name for arg_name, _ in args)})',
        globals={'default_tc_m': default_tc_m} | { arg_name : arg_val for arg_name, arg_val in args },
        label=label,
        description=label,
        sub_label="default torch.compile"
    )

    results.append(unoptimized.blocked_autorange(min_run_time=2))
    results.append(default_compiled.blocked_autorange(min_run_time=2))

    compare = benchmark.Compare(results)
    compare.print()

    return results

    

def benchmark_compile_times():
    q = torch.randn((1024, 64))
    k = torch.randn((1024, 64))
    v = torch.randn((1024, 64))

    results = []

    gern_compile = benchmark.Timer(
        stmt='opt_M = gen(M, torch_to_gern); opt_M(q, k, v)',
        globals={'M': M, 'gen': gen, 'torch_to_gern': torch_to_gern, 'q': q, 'k': k, 'v': v},
        label='compilation',
        description='',
        sub_label=f"gern"
    )
    results.append(gern_compile.timeit(1000))

    default_torch_compile = benchmark.Timer(
        stmt='opt_M = torch.compile(M()); opt_M(q, k, v)',
        globals={'M': M, 'torch': torch, 'q': q, 'k': k, 'v': v},
        label='compilation',
        description='',
        sub_label=f"torch.compile"
    )
    results.append(default_torch_compile.timeit(1000))

    # compare = benchmark.Compare(results)
    # compare.print()

def check_module(M, row_tilings, *args):
    m = M()
    reference = m(*args)

    for row_tiling in row_tilings:
        torch.compiler.reset()
        opt_M = gen(M, torch_to_gern, tile_rows=row_tiling)
        gern_output = opt_M(*args)
        assert(torch.allclose(reference, gern_output, atol=1e-6))

def verify_correctness():
    q = torch.randn((1024, 64))
    k = torch.randn((1024, 64))
    v = torch.randn((1024, 64))

    m = M()
    opt_M = gen(M, torch_to_gern)

    output = opt_M(q, k, v)
    ref = m(q, k, v)
    
    print("OPT_M OUTPUT", output)
    print("NORMAL M OUTPUT", ref)

    print(torch.allclose(ref, output, atol=1e-6))

    q1 = torch.randn((1024, 64))
    k1 = torch.randn((1024, 64))
    v1 = torch.randn((1024, 64))

    output = opt_M(q1, k1, v1)
    ref = m(q1, k1, v1)
    print(torch.allclose(ref, output, atol=1e-6))

    torch.compiler.reset()

    q2 = torch.randn((1024, 32))
    k2 = torch.randn((1024, 32))
    v2 = torch.randn((1024, 32))
    
    output = opt_M(q2, k2, v2)
    ref = m(q2, k2, v2)
    print(torch.allclose(ref, output, atol=1e-6))

def individual_benchmarks():
    class SoftmaxMatmul(torch.nn.Module):
        def forward(self, a, b):
            return torch.nn.functional.softmax(a) @ b
    benchmark_module(SoftmaxMatmul, "softmax (1024, 1024) @ (1024, 64)", [1024], ("a", torch.randn((1024, 1024))), ("b", torch.randn((1024, 64))))

    # all_results = []
    # class Softmax(torch.nn.Module):
    #     def forward(self, arr):
    #         return torch.nn.functional.softmax(arr)
    # row_tilings = [32, 64, 128, 256, 512, 1024]
    
    # arr = torch.randn((1024, 1024))
    # all_results.extend(benchmark_module(Softmax, "softmax (1024, 1024)", row_tilings, ("arr", arr)))
    # check_module(Softmax, row_tilings, arr)

    # class MatMul(torch.nn.Module):
    #     def forward(self, a, b):
    #         return a @ b
    
    # a = torch.randn((1024, 64))
    # b = torch.randn((64, 1024))
    # all_results.extend(benchmark_module(MatMul, "matmul (1024, 64) x (64, 1024)", row_tilings, ("a", a), ("b", b)))
    # check_module(MatMul, row_tilings, a, b)

    # a = torch.randn((1024, 1024))
    # b = torch.randn((1024, 64))
    # all_results.extend(benchmark_module(MatMul, "matmul (1024, 1024) x (1024, 64)", row_tilings, ("a", a), ("b", b)))
    # check_module(MatMul, row_tilings, a, b)

    # compare = benchmark.Compare(all_results)
    # compare.print()    

if __name__ == "__main__":
    verify_correctness()
    # individual_benchmarks() 

    # all_results = []
    # # row_vals = [512, 1024, 1536, 2048]
    # # row_tilings = [[32, 64, 128, 256, 512], [32, 64, 128, 256, 512, 1024], [128, 256, 512, 1536], [512, 1024, 2048]]
    # # col_vals = [32, 64, 96, 128]
    # row_vals = [512]
    # row_tilings=[[32, 64, 128, 256, 512]]
    # col_vals = [64]
    # for rows, row_tiling in zip(row_vals, row_tilings):
    #     for cols in col_vals:
    #         results = benchmark_runtimes(rows, cols, row_tiling)
    #         all_results.append(benchmark.Compare(results))

    # for res in all_results:
    #     res.print()

    # q_32 = torch.randn((1024, 32))
    # k_32 = torch.randn((1024, 32))
    # v_32 = torch.randn((1024, 32))

    # q_64 = torch.randn((1024, 64))
    # k_64 = torch.randn((1024, 64))
    # v_64 = torch.randn((1024, 64))

    # print("GEN")

    # opt_M = gen(M, torch_to_gern)

    # print("CALL WITH 32")
    # opt_M(q_32, k_32, v_32)

   
    
    # print("CALL WITH 32 AGAIN")
    # opt_M(torch.randn((1024, 32)), torch.randn((1024, 32)), torch.randn((1024, 32)))

    # print("CALL WITH 32 AGAIN")
    # opt_M(torch.randn((1024, 32)), torch.randn((1024, 32)), torch.randn((1024, 32)))

    # print("CALL WITH 64")
    # opt_M(torch.randn((1024, 64)), torch.randn((1024, 64)), torch.randn((1024, 64)))

    # print("CALL WITH 64")
    # opt_M(torch.randn((1024, 64)), torch.randn((1024, 64)), torch.randn((1024, 64)))

    

    
