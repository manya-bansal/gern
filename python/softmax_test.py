from gern_py import *
import torch
torch.set_printoptions(precision=6)

sm_in = AbstractDataTypePtr(AnnotMatrixCPU.init("sm_in"))
sm_out = AbstractDataTypePtr(AnnotMatrixCPU.init("sm_out"))

softmax = MatrixSoftmax()


row_val = 10
col_val = 20

l_x = Variable.init("l_x")
l_y = Variable.init("l_y")
y = Variable.init("y")

program = Composable([
    Tile(sm_out["row"], l_x)(
        Tile(sm_out["col"], l_y)(
            softmax(sm_in, sm_out)
        )
    )
])

run = Runner(program)

run.compile(cpuRunner(["matrix"]))

a_tensor = torch.rand(row_val, col_val, dtype=torch.float32)
b_tensor = torch.empty(row_val, col_val, dtype=torch.float32)

a = MatrixCPU.init(a_tensor.data_ptr(), row_val, col_val, a_tensor.stride()[0])
b = MatrixCPU.init(b_tensor.data_ptr(), row_val, col_val, b_tensor.stride()[0])

l_x_val = Int.init(5)
l_y_val = Int.init(col_val)

run.evaluate({
    sm_in.getName(): getAddress(a),
    sm_out.getName(): getAddress(b),
    l_x.getName(): getAddress(l_x_val),
    l_y.getName(): getAddress(l_y_val)
})

ref = a_tensor.softmax(dim=1)
print(torch.allclose(ref, b_tensor))