from generate_torch_compile_pipeline import gen, FnInterface
from gern_py import *
import torch
import operator

torch_to_gern = {
    torch.t: FnInterface(MatrixTranspose),
    operator.matmul: FnInterface(MatrixMultiply, {"shared_len": lambda args: args[0].meta["example_value"].size(dim=1)}),
    operator.truediv: FnInterface(MatrixDivn),
    torch.nn.functional.softmax: FnInterface(MatrixSoftmax)
}

class M(torch.nn.Module):
    def forward(self, a, b):
        return torch.nn.functional.softmax(a / 2) + torch.t(b)

a = torch.randn((10, 10))
b = torch.randn((10, 10))

opt_M = gen(M, torch_to_gern, tile_rows=5)
m = M()
opt_output = opt_M(a, b)
ref = m(a, b)

print("OPT OUTPUT", opt_output)
print("REF OUTPUT", ref)

a1 = torch.randn((10, 10))
b1 = torch.randn((10, 10))

print("OPT_OUTPUT 2", opt_M(a1, b1))
print("REF OUTPUT 2", m(a1, b1))