from gern_py import *
import torch
import math
torch.set_printoptions(precision=6)

q = AbstractDataTypePtr(AnnotMatrixCPU.init("q"))
k = AbstractDataTypePtr(AnnotMatrixCPU.init("k"))
v = AbstractDataTypePtr(AnnotMatrixCPU.init("v"))

kt = AbstractDataTypePtr(AnnotMatrixCPU.init("kt"))
q_kt = AbstractDataTypePtr(AnnotMatrixCPU.init("q_kt"))
sm_in = AbstractDataTypePtr(AnnotMatrixCPU.init("sm_in"))
sm_out = AbstractDataTypePtr(AnnotMatrixCPU.init("sm_out"))

output = AbstractDataTypePtr(AnnotMatrixCPU.init("output"))

l_x = Variable.init("l_x")
l_y = Variable.init("l_y")

q_width = Variable.init("q_width")
q_height = Variable.init("q_height")
sqrt_dk = Variable.init("sqrt_dk", DatatypeClass(Datatype.Float32))

tile_vars = {
    "row": l_x,
    "col": l_y
}

untiled_program = Composable([
    MatrixTranspose(k, kt),
    MatrixMultiply(q, kt, q_kt, {"shared_len": q_width}),
    MatrixDivn(q_kt, sqrt_dk, sm_in),
    MatrixSoftmax(sm_in, sm_out),
    MatrixMultiply(sm_out, v, output, {"shared_len": q_height})
])

annotation = untiled_program.getAnnotation()
pattern = annotation.getPattern()
tileableFields = pattern.getTileableFields()

program = untiled_program

for key, val in tileableFields.items():
    program = Tile(key, tile_vars[key.getMember()])(program)

run = Runner(program)

run.compile(cpuRunner(["matrix"]))

nk_raw = 1024
dk_raw = 64
nk = Int.init(nk_raw)
dk = Int.init(dk_raw)
l_x_val = Int.init(256)
l_y_val = Int.init(dk_raw)
sqrt_dk_val = Float.init(math.sqrt(dk_raw))

in_q_tensor = torch.rand(nk_raw, dk_raw, dtype=torch.float32)
in_k_tensor = torch.rand(nk_raw, dk_raw, dtype=torch.float32)
in_v_tensor = torch.rand(nk_raw, dk_raw, dtype=torch.float32)
out_tensor = torch.empty(nk_raw, dk_raw, dtype=torch.float32)

in_q = MatrixCPU.init(in_q_tensor.data_ptr(), *in_q_tensor.size(), in_q_tensor.stride()[0])
in_k = MatrixCPU.init(in_k_tensor.data_ptr(), *in_k_tensor.size(), in_k_tensor.stride()[0])
in_v = MatrixCPU.init(in_v_tensor.data_ptr(), *in_v_tensor.size(), in_v_tensor.stride()[0])
out = MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])


run.evaluate({
    q.getName(): getAddress(in_q),
    k.getName(): getAddress(in_k),
    v.getName(): getAddress(in_v),
    sqrt_dk.getName(): getAddress(sqrt_dk_val),
    output.getName(): getAddress(out),
    l_x.getName(): getAddress(l_x_val),
    l_y.getName(): getAddress(l_y_val),
    q_width.getName(): getAddress(dk),
    q_height.getName(): getAddress(nk)
})

def attention(q, k, v):
    return torch.nn.functional.softmax((q @ torch.t(k)) / math.sqrt(q.size(dim=1)), dim=1) @ v

ref = attention(in_q_tensor, in_k_tensor, in_v_tensor)

print(torch.allclose(ref, out_tensor))