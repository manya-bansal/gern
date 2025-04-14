from gern_py import *
import torch
import math
torch.set_printoptions(precision=6)

q = AbstractDataTypePtr(AnnotMatrixCPU4Dim.init("q"))
k = AbstractDataTypePtr(AnnotMatrixCPU4Dim.init("k"))
v = AbstractDataTypePtr(AnnotMatrixCPU4Dim.init("v"))

output = AbstractDataTypePtr(AnnotMatrixCPU4Dim.init("output"))

l_w = Variable.init("l_w")
l_x = Variable.init("l_x")
l_y = Variable.init("l_y")
l_z = Variable.init("l_z")

q_width = Variable.init("q_width")
q_height = Variable.init("q_height")
sqrt_dk = Variable.init("sqrt_dk", DatatypeClass(Datatype.Float32))

attention = MatrixAttention4D()
attention.setBindings({"height": q_height, "width": q_width})

program = Composable([
    Tile(output["i_dim"], l_w)(
        Tile(output["j_dim"], l_x)(
            Tile(output["k_dim"], l_y)(
                Tile(output["l_dim"], l_z)(
                    attention(q, k, v, output)
                )
            )
        )
    )
])

run = Runner(program)

run.compile(cpuRunner(["matrix"]))

nk_raw = 1024
dk_raw = 64
nk = Int.init(nk_raw)
dk = Int.init(dk_raw)
l_w_val = Int.init(1)
l_x_val = Int.init(12)
l_y_val = Int.init(256)
l_z_val = Int.init(dk_raw)

in_q_tensor = torch.rand(1, 12, nk_raw, dk_raw, dtype=torch.float32)
in_k_tensor = torch.rand(1, 12, nk_raw, dk_raw, dtype=torch.float32)
in_v_tensor = torch.rand(1, 12, nk_raw, dk_raw, dtype=torch.float32)
out_tensor = torch.empty(1, 12, nk_raw, dk_raw, dtype=torch.float32)

in_q = MatrixCPU4Dim.init(in_q_tensor.data_ptr(), *in_q_tensor.size(), *in_q_tensor.size()[1:])
in_k = MatrixCPU4Dim.init(in_k_tensor.data_ptr(), *in_k_tensor.size(), *in_k_tensor.size()[1:])
in_v = MatrixCPU4Dim.init(in_v_tensor.data_ptr(), *in_v_tensor.size(), *in_v_tensor.size()[1:])
out = MatrixCPU4Dim.init(out_tensor.data_ptr(), *out_tensor.size(), *out_tensor.size()[1:])
print(*in_q_tensor.size(), *in_q_tensor.size()[1:])

run.evaluate({
    q.getName(): getAddress(in_q),
    k.getName(): getAddress(in_k),
    v.getName(): getAddress(in_v),
    output.getName(): getAddress(out),
    l_w.getName(): getAddress(l_w_val),
    l_x.getName(): getAddress(l_x_val),
    l_y.getName(): getAddress(l_y_val),
    l_z.getName(): getAddress(l_z_val),
    q_width.getName(): getAddress(dk),
    q_height.getName(): getAddress(nk)
})

def attention(q, k, v):
    return torch.nn.functional.softmax((q @ torch.t(k)) / math.sqrt(q.size(dim=1)), dim=1) @ v

ref = torch.nn.functional.scaled_dot_product_attention(in_q_tensor, in_k_tensor, in_v_tensor)

print(torch.allclose(ref, out_tensor))