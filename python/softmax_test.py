from gern_py import *
import torch
torch.set_printoptions(precision=6)

sm_in = AbstractDataTypePtr(AnnotMatrixCPU.init("sm_in"))
sm_out = AbstractDataTypePtr(AnnotMatrixCPU.init("sm_out"))

row_val = 10
col_val = 20

l_x = Variable.init("l_x")
l_y = Variable.init("l_y")
y = Variable.init("y")

program = Composable([
    Tile(sm_out["row"], l_x)(
        MatrixSoftmax(sm_in, sm_out, {
            "y": y.bind(0),
            "l_y": l_y.bind(col_val)
        })
    )
])

run = Runner(program)

run.compile(cpuRunner(["matrix"]))

a_tensor = torch.rand(row_val, col_val, dtype=torch.float32)
b_tensor = torch.empty(row_val, col_val, dtype=torch.float32)

a = MatrixCPU.init(a_tensor.data_ptr(), row_val, col_val, a_tensor.stride()[0])
b = MatrixCPU.init(b_tensor.data_ptr(), row_val, col_val, b_tensor.stride()[0])

l_x_val = Int.init(5)

run.evaluate({
    sm_in.getName(): getAddress(a),
    sm_out.getName(): getAddress(b),
    l_x.getName(): getAddress(l_x_val)
})

ref = a_tensor.softmax(dim=1)
print(torch.allclose(ref, b_tensor))