import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from gern_py import *

from typing import List
import time
import math
import subprocess
import ctypes
import os

class FnInterface:
    def __init__(self, fn, extra_args={}):
        self.fn = fn
        self.extra_args = extra_args

def gen(M, torch_to_gern, *args):
    variables = []

    generated_runner = None

    input_name_to_arg_idx = {}
    input_adt_ptr_to_arg_idx = {}

    out_size = None
    output_adt_ptr = None

    def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("custom backend called with FX graph:")
        print(example_inputs)
        gm.graph.print_tabular()
        print("GM FORWARD")
        def custom_callable(*args):
            # print("ARGS", args)
            nonlocal generated_runner
            if generated_runner is None:
                # print("COMPILING RUNNER")
                # ShapeProp(gm).propagate(*args)
                out = None
                out = gm.forward(*args)

                env = {}

                dummy_var_idx = {}

                input_vals = {}


                fn_calls = []

                for i, node in enumerate(gm.graph.find_nodes(op="placeholder")):
                    if isinstance(example_inputs[i], torch.Tensor):
                        env[node.name] = AbstractDataTypePtr(AnnotMatrixCPU.init(node.name))
                        input_name_to_arg_idx[node.name] = i
                    input_vals[node.name] = args[i]
                    print("i", i, "node", node, "args[i]", args[i])
                    
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue

                    print("node.target", node.target)
                    print("node.args", node.args)
                    print("node.name", node.name)


                    if node.target in torch_to_gern:
                        fn_interface = torch_to_gern[node.target]

                        evaled_args = []
                        for i, arg in enumerate(node.args):
                            if isinstance(arg, torch.fx.Node):
                                evaled_args.append(env[arg.name])
                            else:
                                var_name = node.name + "_" + str(i)
                                var = Variable.init(var_name, DatatypeClass(Datatype.Float32))
                                variables.append((var, Float.init(arg)))
                                evaled_args.append(var)

                        result_arg = AbstractDataTypePtr(AnnotMatrixCPU.init(node.name))
                        env[node.name] = result_arg

                        specialize_dict = {}
                        for specialize_arg in fn_interface.extra_args:
                            var_name = node.name + "_" + specialize_arg
                            var = Variable.init(var_name)
                            variables.append((var, Int.init(fn_interface.extra_args[specialize_arg](node.args))))
                            specialize_dict[specialize_arg] = var

                        fn_calls.append(fn_interface.fn(*evaled_args, result_arg, specialize_dict))
                    else:
                        evaled_args = []
                        for arg in node.args:
                            evaled_args.append(input_vals[arg.name])

                        evaled_val = node.target(*evaled_args)
                        var = Variable.init(node.name, DatatypeClass(Datatype.Float32))
                        variables.append((var, Float.init(evaled_val)))
                        env[node.name] = var

                output_node = gm.graph.find_nodes(op="output")[0].args[0][0]

                nonlocal out_size
                out_size = output_node.meta['example_value'].size()

                out_tensor = torch.empty(list(map(int, out_size)))

                # out_tensor = torch.empty(size=output_node.meta['example_value'].size())

                l_x = Variable.init("l_x")
                l_y = Variable.init("l_y")

                nonlocal output_adt_ptr
                output_adt_ptr = env[output_node.name]

                program = Composable([
                    Tile(output_adt_ptr["row"], l_x)(
                        Tile(output_adt_ptr["col"], l_y)(
                            *fn_calls
                        )
                    )
                ])

                generated_runner = Runner(program)

                generated_runner.compile(cpuRunner(["matrix"]))

                l_x_val = Int.init(512)
                l_y_val = Int.init(64)

                variables.append((l_x, l_x_val))
                variables.append((l_y, l_y_val))

                nonlocal input_adt_ptr_to_arg_idx
                input_adt_ptr_to_arg_idx = {
                    env[input_arg_name]: input_arg_idx
                    for input_arg_name, input_arg_idx in input_name_to_arg_idx.items()
                }


                generated_runner.evaluate({
                    var.getName(): getAddress(var_val)
                    for var, var_val in variables
                } | {
                    input_adt_ptr.getName(): getAddress(
                        MatrixCPU.init(args[i].data_ptr(), *args[i].size(), args[i].stride()[0])
                    )
                    for input_adt_ptr, i in input_adt_ptr_to_arg_idx.items()
                } | {
                    output_adt_ptr.getName(): getAddress(
                        MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                    )
                })

                return (out_tensor,)
                # return gm.forward(*args)
            else:
                # print("REUSING GENERATED RUNNER")

                out_tensor = torch.empty(list(map(int, out_size)))

                args = {
                    var.getName(): getAddress(var_val)
                    for var, var_val in variables
                } | {
                    input_adt_ptr.getName(): getAddress(
                        MatrixCPU.init(args[i].data_ptr(), *args[i].size(), args[i].stride()[0])
                    )
                    for input_adt_ptr, i in input_adt_ptr_to_arg_idx.items()
                } | {
                    output_adt_ptr.getName(): getAddress(
                        MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                    )
                }

                generated_runner.evaluate(args)

                return (out_tensor,)
        return custom_callable
    
    # opt_M = torch.compile(M(), backend=custom_backend, dynamic=True)
    opt_M = torch.compile(M(), backend=custom_backend)
    return opt_M
   
    