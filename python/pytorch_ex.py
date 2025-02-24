import torch
import torch.fx
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

if __name__ == "__main__":
    opt_M = gen(M, torch_to_gern, torch.randn((1024, 64)), torch.randn((1024, 64)), torch.randn((1024, 64)))

    q = torch.randn((1024, 64))
    k = torch.randn((1024, 64))
    v = torch.randn((1024, 64))

    m = M()

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

    # for i in range(10):
    #     opt_M(q, k, v)

    # for i in range(10):
    #     time_start = time.perf_counter()
    #     m(q, k, v)
    #     time_end = time.perf_counter()
    #     print(time_end - time_start)

    default_tc_m = torch.compile(M())

    import torch.utils.benchmark as benchmark

    optimized = benchmark.Timer(
        setup='opt_M(q, k, v)',
        stmt='opt_M(q, k, v)',
        globals={'opt_M': opt_M, 'q': q, 'k': k, 'v': v}
    )
    
    unoptimized = benchmark.Timer(
        stmt='m(q, k, v)',
        globals={'m': m, 'q': q, 'k': k, 'v': v}
    )

    default_compiled = benchmark.Timer(
        setup='default_tc_m(q, k, v)',
        stmt='default_tc_m(q, k, v)',
        globals={'default_tc_m': default_tc_m, 'q': q, 'k': k, 'v': v}
    )

    print(optimized.timeit(1000))
    print(unoptimized.timeit(1000))
    print(default_compiled.timeit(1000))
