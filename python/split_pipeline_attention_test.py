from generate_torch_compile_pipeline import gen, FnInterface
from gern_py import *
import torch
import operator

torch_to_gern = {
    torch.t: FnInterface(MatrixTranspose),
    operator.matmul: FnInterface(MatrixMultiply, {"shared_len": lambda args: args[0].meta["example_value"].size(dim=-1)}),
    operator.truediv: FnInterface(MatrixDivn),
    torch.nn.functional.softmax: FnInterface(MatrixSoftmax),
    torch.nn.functional.scaled_dot_product_attention: FnInterface(MatrixAttention, {"height": lambda args: args[0].meta["example_value"].size(dim=-2), "width": lambda args: args[0].meta["example_value"].size(dim=-1)})
}

class M(torch.nn.Module):
    def forward(self, q, k, v, c):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v) + c

q = torch.randn((1, 12, 10, 10))
k = torch.randn((1, 12, 10, 10))
v = torch.randn((1, 12, 10, 10))
c = torch.randn((1, 12, 10, 10))

m = M()
ref = m(q, k, v, c)


opt_M = gen(M, torch_to_gern, tile_rows=5)
opt_m = opt_M()
opt_output = opt_m(q, k, v, c)

print("OPT OUTPUT", opt_output)
print("REF OUTPUT", ref)
print("OUTPUTS MATCH", torch.allclose(opt_output, ref, atol=1e-6))

# a1 = torch.randn((10, 10))
# b1 = torch.randn((10, 10))

# opt_output2 = opt_m(a1, b1)
# ref2 = m(a1, b1)

# print("OPT_OUTPUT 2", opt_output2)
# print("REF OUTPUT 2", ref2)
# print("OUTPUTS MATCH", torch.allclose(opt_output2, ref2))