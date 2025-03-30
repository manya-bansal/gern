from gern_py import *

inputDS = AbstractDataTypePtr(AnnotMatrixCPU.init("input_con"))
tmp = AbstractDataTypePtr(AnnotMatrixCPU.init("tmp"))
outputDS = AbstractDataTypePtr(AnnotMatrixCPU.init("output_con"))

add1 = MatrixAddCPU()
add2 = MatrixAddCPU()

l_x = Variable.init("l_x")
l_y = Variable.init("l_y")
l_x_bound = l_x.bind(5)
l_y_bound = l_y.bind(5)

program = Tile(outputDS["row"], l_x_bound)(
        Tile(outputDS["col"], l_y_bound)(
            add1(inputDS, tmp),
            add2(tmp, outputDS)
        )
    )


run = Runner(program)
print(run)

run.compile(cpuRunner(["matrix"]))

row_val = 10
col_val = 10
a = MatrixCPU.init(row_val, col_val, col_val)
a.vvals(2.0)
b = MatrixCPU.init(row_val, col_val, col_val)

l_x_val = Int.init(5)
l_y_val = Int.init(5)

run.evaluate({
    inputDS.getName(): getAddress(a),
    outputDS.getName(): getAddress(b),
    # l_x.getName(): getAddress(l_x_val),
    # l_y.getName(): getAddress(l_y_val)
})

print(b)