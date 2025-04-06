import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from gern_py import *

from typing import List
import itertools
import time
import math
import subprocess
import ctypes
import os

class FnInterface:
    def __init__(self, fn, extra_args={}):
        self.fn = fn
        self.extra_args = extra_args


def function_call_fn(runner, inp, output_adt_ptr, out_size, input_adt_ptrs, *args):
    args = [torch.Tensor.contiguous(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
    # print("CALLING EVALUATE")
    # print("INPUT ADT PTRS", input_adt_ptrs)

    # print("ARGS", args)

    # for arg in args:
    #     print(arg.data_ptr())
    
    final_out = torch.zeros(list(map(int, out_size)))

    for idx in itertools.product(*[range(s) for s in out_size[:-2]]):
        input_args = {
            input_adt_ptr.getName(): getAddress(
                MatrixCPU.init(input_val[idx].data_ptr(), *input_val[idx].size(), input_val[idx].stride()[0])
                if isinstance(input_val, torch.Tensor)
                else Float.init(input_val)
            )
            for input_adt_ptr, input_val in zip(input_adt_ptrs, args)
        }

        out_tensor = final_out[idx]

        evaluate_args = inp | input_args | {
                                            output_adt_ptr.getName(): getAddress(
                                                MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                                            )
                                        }
        for arg in evaluate_args:
            print("ARG NAME", arg)
        runner.evaluate(evaluate_args)
    return final_out

def gen(M, torch_to_gern, *args, tile_rows=512):
    def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        graph_edited = False
        variables = []

        # print("custom backend called with FX graph:")
        # print(example_inputs)
        # gm.graph.print_tabular()
        def custom_callable(*args):
            # print("ARGS", args)
            # print("EXAMPLE INPUTS", example_inputs)
            nonlocal graph_edited
            if not graph_edited: 
                graph_edited = True

                # a dict of node names to AbstractDataTypePtrs
                adt_pointers = {}
                fn_calls = []

                # for i, node in enumerate(gm.graph.find_nodes(op="placeholder")):
                #     if isinstance(example_inputs[i], torch.Tensor):
                #         env[node.name] = AbstractDataTypePtr(AnnotMatrixCPU.init(node.name))
                #         input_name_to_arg_idx[node.name] = i
                #     input_vals[node.name] = args[i]
                    # print("i", i, "node", node, "args[i]", args[i])
                
                def convert_node_to_gern(node):
                    print("CONVERT NODE TO GERN", node)
                    if not isinstance(node, torch.fx.node.Node):
                        return
                    
                    output_node = node
                    print("META", node, output_node.meta)
                    out_size = output_node.meta['example_value'].size()
                    two_dim_out_size = out_size[-2:]
                    print("TWO DIM OUT SIZE", two_dim_out_size)
                    # out_tensor = torch.empty(list(map(int, out_size)))

                    # this is not a gern function
                    if output_node.name not in adt_pointers:
                        return

                    output_adt_ptr = adt_pointers[output_node.name]

                    l_x = Variable.init("l_x")
                    l_y = Variable.init("l_y")

                    relevant_outputs = [output_adt_ptr]
                    relevant_fn_calls = set()
                    out_list_changed = True
                    relevant_output_nodes = [output_node]
                    intermediates = []

                    print(fn_calls)
                    while out_list_changed:
                        orig_fn_call_sz = len(relevant_fn_calls)
                        out_list_changed = False
                        for fn_call, inputs, input_nodes, result, extra_vars in fn_calls:
                            if fn_call in relevant_fn_calls:
                                continue
                            print("fn call", fn_call, "result", result, "relevant outputs", relevant_outputs, result in relevant_outputs)
                            if result in relevant_outputs:
                                print("HERE")
                                intermediates.append(result)
                                for inp, inp_node in zip(inputs, input_nodes):
                                    print("ADD INP", inp)
                                    relevant_outputs.append(inp)
                                    relevant_output_nodes.append(inp_node)
                                relevant_fn_calls.add(fn_call)
                        if len(relevant_fn_calls) > orig_fn_call_sz:
                            out_list_changed = True

                    if len(relevant_fn_calls) == 0:
                        return

                    relevant_inputs = []
                    relevant_input_nodes = []

                    for arg_adt_ptr, arg_node in zip(relevant_outputs, relevant_output_nodes):
                        if arg_adt_ptr not in intermediates:
                            relevant_inputs.append(arg_adt_ptr)
                            relevant_input_nodes.append(arg_node)

                    print("ARG TARGET", node.target)
                    print("ARG ARGS", node.args)
                    print("ARG KWARGS", node.kwargs)
                
                    print("RELEVANT OUTPUTS", relevant_outputs)
                    print("RELEVANT FN CALLS", relevant_fn_calls)

                    filtered_fn_calls = [fn_call for fn_call, inputs, inp_nodes, result, extra_vars in fn_calls if fn_call in relevant_fn_calls]
                    filtered_variable_names = [var for fn_call, inputs, inp_nodes, result, extra_vars in fn_calls if fn_call in relevant_fn_calls for var in inputs if isinstance(var, Variable)]

                    for fn_call, inputs, inp_nodes, result, extra_vars in fn_calls:
                        if fn_call in relevant_fn_calls:
                            for var in extra_vars:
                                filtered_variable_names.append(var)

                    filtered_variables = [(var, var_val) for var, var_val in variables if var in filtered_variable_names]
                    print("FILTERED VARIABLES", filtered_variables)

                    program = Composable([
                        Tile(output_adt_ptr["row"], l_x)(
                            Tile(output_adt_ptr["col"], l_y)(
                                *filtered_fn_calls
                            )
                        )
                    ])

                    generated_runner = Runner(program)

                    print("RUNNER COMPILE")
                    generated_runner.compile(cpuRunner(["matrix"]))
                    print(filtered_fn_calls)
                    print("FINISH COMPILING")

                    l_x_val = Int.init(tile_rows)
                    l_y_val = Int.init(two_dim_out_size[1])

                    filtered_variables.append((l_x, l_x_val))
                    filtered_variables.append((l_y, l_y_val))

                    print("FILTERED VARIABLES")
                    for var, var_val in filtered_variables:
                        print(var.getName(), var_val)

                    gern_args = {
                        var.getName(): getAddress(var_val)
                        for var, var_val in filtered_variables
                    } 
                    print("GERN ARGS")
                    for var, var_val in filtered_variables:
                        print("Var", var)
                        print("Val", var_val)
                    
                    with gm.graph.inserting_before(None):
                        print("PRINTING GM GRAPH" , gm.graph)
                        setattr(gm, node.name + '_generated_runner', generated_runner)
                        setattr(gm, node.name + '_gern_args', gern_args)
                        setattr(gm, node.name + '_output_adt_ptr', output_adt_ptr)
                        setattr(gm, node.name + "_out_size", output_node.meta['example_value'].size())
                        setattr(gm, node.name + '_inputs', relevant_inputs)

                        runner_node = gm.graph.create_node('get_attr', node.name + '_generated_runner')
                        capsules_node = gm.graph.create_node('get_attr', node.name + '_gern_args')
                        output_adt_ptr_node = gm.graph.create_node('get_attr', node.name + '_output_adt_ptr')
                        out_size_node = gm.graph.create_node('get_attr', node.name + '_out_size')
                        input_node = gm.graph.create_node('get_attr', node.name + '_inputs')

                        print("PRINTING GM GRAPH" , gm.graph)

                    print("RELEVANT OUTPUT NODES", len(relevant_output_nodes), relevant_output_nodes)
                    with gm.graph.inserting_after(node):
                        new_node = gm.graph.create_node("call_function", function_call_fn, args=(runner_node, capsules_node, output_adt_ptr_node, out_size_node, input_node, *relevant_input_nodes), name="replaced_" + node.name)
                        node.replace_all_uses_with(new_node)
                        gm.graph.erase_node(node)
                    
                for node in gm.graph.nodes:
                    if node.op != "call_function" and node.op != "call_method":
                        continue

                    print("node.target", node.target)
                    print("node.args", node.args)
                    print("node.name", node.name)


                    if node.target in torch_to_gern:
                        fn_interface = torch_to_gern[node.target]

                        abstract_args = []
                        arg_nodes = []
                        print("NODE TARGET", node.target)
                        for arg in node.args:
                            if isinstance(arg, torch.fx.Node):
                                if arg.name not in adt_pointers:
                                    adt_pointers[arg.name] = AbstractDataTypePtr(AnnotMatrixCPU.init(arg.name))
                                    print("ASSIGNING ADT POINTER", arg.name)
                                abstract_args.append(adt_pointers[arg.name])
                            else:
                                var = Variable.init(node.name + "_variable_arg", DatatypeClass(Datatype.Float32))
                                variables.append((var, Float.init(arg)))
                                abstract_args.append(var)
                            arg_nodes.append(arg)

                        result_arg = AbstractDataTypePtr(AnnotMatrixCPU.init(node.name))
                        print("ASSIGNING ADT POINTER", node.name)
                        adt_pointers[node.name] = result_arg

                        specialize_dict = {}
                        extra_vars = []
                        for specialize_arg_name, specialize_arg_val_fn in fn_interface.extra_args.items():
                            var_name = node.name + "_" + specialize_arg_name
                            var = Variable.init(var_name)
                            variables.append((var, Int.init(specialize_arg_val_fn(node.args))))
                            extra_vars.append(var)
                            specialize_dict[specialize_arg_name] = var

                        fn_instance = fn_interface.fn()
                        fn_instance.setBindings(specialize_dict)
                        fn_calls.append((fn_instance(*abstract_args, result_arg), abstract_args, arg_nodes, result_arg, extra_vars))
                    else:
                        # need to split pipeline 
                        print(node.meta)
                        for arg in node.args:
                            convert_node_to_gern(arg) 

                graph_output = gm.graph.find_nodes(op="output")[0]
                print("GRAPH OUTPUT", graph_output)
                print("GRAPH OUTPUT ARGS", graph_output.args)
                convert_node_to_gern(graph_output.args[0][0])

                gm.graph.eliminate_dead_code()
                print("PRINTING FINAL GM GRAPH" , gm.graph)
                gm.recompile()
                return gm.forward(*args)

            else:
                print("REUSING EDITED GRAPH")

                return gm.forward(*args)
        return custom_callable
    
    # opt_M = torch.compile(M(), backend=custom_backend, dynamic=True)
    opt_M = torch.compile(M, backend=custom_backend)
    return opt_M
   
    