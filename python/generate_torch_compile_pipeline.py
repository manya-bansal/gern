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

variables = []

def function_call_fn(runner, inp, output_adt_ptr, out_size, input_adt_ptrs, orig_arg, *args):
    print("CALLING EVALUATE")
    print("INPUT ADT PTRS", input_adt_ptrs)
    print(variables)
    for var, var_val in variables:
        print("VAR GETNAME", var.getName())
        print("GETADDRESS", getAddress(var_val))
    print("ARGS", args)

    for arg in args:
        print(arg.data_ptr())
    
    final_out = torch.zeros(list(map(int, out_size)))
    if len(out_size) == 4:
        for i in range(out_size[0]):
            for j in range(out_size[1]):
                print("HERE 1")
                input_args = {input_adt_ptr.getName(): getAddress(
                    MatrixCPU.init(input_tensor[i][j].data_ptr(), *input_tensor[i][j].size(), input_tensor[i][j].stride()[0])
                ) for input_adt_ptr, input_tensor in zip(input_adt_ptrs, args)}
                print("HERE 2")

                print(out_size)
                # out_tensor = torch.zeros(5, 5)
                out_tensor = final_out[i, j, :, :]
                # print(out_tensor.size())
                # print("OUT TENSOR DATA PTR", out_tensor.data_ptr())
                # print("OUT TENSOR STRIDE", out_tensor.stride())
                # print("OUT TENSOR VALUES", out_tensor)
                print("HERE 3")

                evaluate_args = inp | input_args | {
                                            output_adt_ptr.getName(): getAddress(
                                                MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                                            )
                                        }
                print("HERE 4")

                print("EVALUATE ARGS", evaluate_args)
                runner.evaluate(evaluate_args)
                # print("OUT TENSOR VALS", out_tensor)
        # print("FINAL OUTPUT SIZE", final_out.size())
        # print("FINAL OUT VALS", final_out)
        # print("ACTUAL VAL", orig_arg)
        print("FINAL ACTUAL MATCH", torch.allclose(final_out, orig_arg, atol=1e-6))
        return final_out
    else:
        input_args = {input_adt_ptr.getName(): getAddress(
                    MatrixCPU.init(input_tensor.data_ptr(), *input_tensor.size(), input_tensor.stride()[0])
                ) for input_adt_ptr, input_tensor in zip(input_adt_ptrs, args)}
        evaluate_args = inp | input_args | {
                                        output_adt_ptr.getName(): getAddress(
                                            MatrixCPU.init(final_out.data_ptr(), *final_out.size(), final_out.stride()[0])
                                        )
                                    }
        runner.evaluate(evaluate_args)
        print("FINAL OUT VALS", final_out)
        print("ACTUAL VAL", orig_arg)
        print("FINAL ACTUAL MATCH", torch.allclose(final_out, orig_arg, atol=1e-6))
        return final_out

def gen(M, torch_to_gern, *args, tile_rows=512):
    def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        generated_runner = None
        variables.clear()

        input_name_to_arg_idx = {}
        input_adt_ptr_to_arg_idx = {}

        out_size = None
        output_adt_ptr = None

        pipeline_sections = []
        python_computations = []
        # print("custom backend called with FX graph:")
        # print(example_inputs)
        # gm.graph.print_tabular()
        def custom_callable(*args):
            # print("ARGS", args)
            # print("EXAMPLE INPUTS", example_inputs)
            nonlocal generated_runner
            if generated_runner is None:
                # print("COMPILING RUNNER")
                # ShapeProp(gm).propagate(*args)
                # out = None
                # out = gm.forward(*args)

                env = {}

                dummy_var_idx = {}

                input_vals = {}


                fn_calls = []

                # for i, node in enumerate(gm.graph.find_nodes(op="placeholder")):
                #     if isinstance(example_inputs[i], torch.Tensor):
                #         env[node.name] = AbstractDataTypePtr(AnnotMatrixCPU.init(node.name))
                #         input_name_to_arg_idx[node.name] = i
                #     input_vals[node.name] = args[i]
                    # print("i", i, "node", node, "args[i]", args[i])
                    
                for node in gm.graph.nodes:
                    if node.op != "call_function" and node.op != "call_method":
                        continue

                    print("node.target", node.target)
                    print("node.args", node.args)
                    print("node.name", node.name)


                    if node.target in torch_to_gern:
                        fn_interface = torch_to_gern[node.target]

                        evaled_args = []
                        arg_nodes = []
                        print("NODE TARGET", node.target)
                        for i, arg in enumerate(node.args):
                            if isinstance(arg, torch.fx.Node):
                                env[arg.name] = AbstractDataTypePtr(AnnotMatrixCPU.init(arg.name))
                                evaled_args.append(env[arg.name])
                            else:
                                var_name = node.name + "_" + str(i)
                                var = Variable.init(var_name, DatatypeClass(Datatype.Float32))
                                variables.append((var, Float.init(arg)))
                                evaled_args.append(var)
                            arg_nodes.append(arg)

                        result_arg = AbstractDataTypePtr(AnnotMatrixCPU.init(node.name))
                        env[node.name] = result_arg

                        specialize_dict = {}
                        extra_vars = []
                        for specialize_arg in fn_interface.extra_args:
                            var_name = node.name + "_" + specialize_arg
                            var = Variable.init(var_name)
                            variables.append((var, Int.init(fn_interface.extra_args[specialize_arg](node.args))))
                            extra_vars.append(var)
                            specialize_dict[specialize_arg] = var

                        fn_instance = fn_interface.fn()
                        fn_instance.setBindings(specialize_dict)
                        fn_calls.append((fn_instance(*evaled_args, result_arg), evaled_args, arg_nodes, result_arg, extra_vars))
                    else:
                        # need to split pipeline 
                        evaled_args_idx = []
                        print(node.meta)
                        for arg in node.args:
                            if not isinstance(arg, torch.fx.node.Node):
                                continue
                            print("SPLIT PIPELINE ARG", arg)
                            print(type(arg))
                            
                            output_node = arg
                            out_size = output_node.meta['example_value'].size()
                            two_dim_out_size = out_size[-2:]
                            print("TWO DIM OUT SIZE", two_dim_out_size)
                            # out_tensor = torch.empty(list(map(int, out_size)))

                            if output_node.name not in env:
                                env[output_node.name] = AbstractDataTypePtr(AnnotMatrixCPU.init(output_node.name))
                            output_adt_ptr = env[output_node.name]

                            l_x = Variable.init("l_x")
                            l_y = Variable.init("l_y")

                            relevant_outputs = [output_adt_ptr]
                            relevant_output_names = {output_adt_ptr.getName()}
                            relevant_fn_calls = set()
                            out_list_changed = True
                            relevant_output_nodes = [output_node]

                            print(fn_calls)
                            while out_list_changed:
                                orig_fn_call_sz = len(relevant_fn_calls)
                                out_list_changed = False
                                for fn_call, inputs, input_nodes, result, extra_vars in fn_calls:
                                    if fn_call in relevant_fn_calls:
                                        continue
                                    if result in relevant_outputs:
                                        for inp, inp_node in zip(inputs, input_nodes):
                                            print("ADD INP", inp)
                                            relevant_outputs.append(inp)
                                            relevant_output_names.add(inp.getName())
                                            relevant_output_nodes.append(inp_node)
                                        relevant_fn_calls.add(fn_call)
                                if len(relevant_fn_calls) > orig_fn_call_sz:
                                    out_list_changed = True

                            

                            if len(relevant_fn_calls) == 0:
                                continue

                            print("ARG TARGET", arg.target)
                            print("ARG ARGS", arg.args)
                            print("ARG KWARGS", arg.kwargs)
                            kwarg = arg.kwargs['attn_mask']
                        
                            print("RELEVANT OUTPUTS", relevant_outputs)
                            print("RELEVANT FN CALLS", relevant_fn_calls)

                            filtered_fn_calls = [fn_call for fn_call, inputs, inp_nodes, result, extra_vars in fn_calls if fn_call in relevant_fn_calls]
                            filtered_variable_names = [var for fn_call, inputs, inp_nodes, result, extra_vars in fn_calls if fn_call in relevant_fn_calls for var in inputs if isinstance(var, Variable)]

                            for fn_call, inputs, inp_nodes, result, extra_vars in fn_calls:
                                if fn_call in relevant_fn_calls:
                                    for var in extra_vars:
                                        filtered_variable_names.append(var)

                            filtered_variables = [(var, var_val) for var, var_val in variables if var in filtered_variable_names]
                            # filtered_inputs = {key: val for key, val in input_name_to_arg_idx.items() if key in relevant_output_names}
                            print("FILTERED VARIABLES", filtered_variables)
                            # print(filtered_inputs)

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

                            # input_adt_ptr_to_arg_idx = {
                            #     env[input_arg_name]: input_arg_idx
                            #     for input_arg_name, input_arg_idx in filtered_inputs.items()
                            # }

                            # print("OUT TENSOR SIZE", out_tensor.size())
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
                            
                            # | {
                            #     output_adt_ptr.getName(): getAddress(
                            #         MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                            #     )
                            # }

                            '''
                            | {
                                input_adt_ptr.getName(): getAddress(
                                    MatrixCPU.init(args[i].data_ptr(), *args[i].size(), args[i].stride()[0])
                                )
                                for input_adt_ptr, i in input_adt_ptr_to_arg_idx.items()
                            } 
                            '''
                            with gm.graph.inserting_before(None):
                                print("PRINTING GM GRAPH" , gm.graph)
                                setattr(gm, arg.name + '_generated_runner', generated_runner)
                                setattr(gm, arg.name + '_gern_args', gern_args)
                                setattr(gm, arg.name + '_output_adt_ptr', output_adt_ptr)
                                setattr(gm, arg.name + "_out_size", output_node.meta['example_value'].size())
                                setattr(gm, arg.name + '_inputs', relevant_outputs[1:])
                                # setattr(gm, arg.name + '_filtered_variables', filtered_variables)

                                runner_node = gm.graph.create_node('get_attr', arg.name + '_generated_runner')
                                capsules_node = gm.graph.create_node('get_attr', arg.name + '_gern_args')
                                output_adt_ptr_node = gm.graph.create_node('get_attr', arg.name + '_output_adt_ptr')
                                out_size_node = gm.graph.create_node('get_attr', arg.name + '_out_size')
                                input_node = gm.graph.create_node('get_attr', arg.name + '_inputs')
                                # filtered_variables_node = gm.graph.create_node('get_attr', arg.name + '_filtered_variables')
                                gm.recompile()
                                print("PRINTING GM GRAPH" , gm.graph)

                            print("RELEVANT OUTPUT NODES", len(relevant_output_nodes), relevant_output_nodes)
                            with gm.graph.inserting_after(arg):
                                new_arg = gm.graph.create_node("call_function", function_call_fn, args=(runner_node, capsules_node, output_adt_ptr_node, out_size_node, input_node, arg, *relevant_output_nodes[1:]), name="replaced_" + arg.name)
                                arg.replace_all_uses_with(new_arg)
                                new_arg.replace_input_with(new_arg, arg)
                                print(gm.graph)

                            # pipeline_sections.append((generated_runner, gern_args, out_tensor))
                            # pipeline_idx = len(pipeline_sections) - 1
                            # evaled_args_idx.append(pipeline_idx)

                        #     print(python_computations)
                        #     print(pipeline_sections)
                        # python_computations.append((node.target, evaled_args_idx))

                        # evaled_val = node.target(*evaled_args)
                        # var = Variable.init(node.name, DatatypeClass(Datatype.Float32))
                        # variables.append((var, Float.init(evaled_val)))
                        # env[node.name] = var

                ## SPLIT PIPELINES
                # print(python_computations)
                # print(pipeline_sections)
                # out = None

                # out_tensors = []
                # for generated_runner, gern_args, out_tensor in pipeline_sections:
                #     print("GERN ARGS", gern_args)
                #     generated_runner.evaluate(gern_args)
                #     out_tensors.append(out_tensor)

                # for comp_fn, args in python_computations:
                #     out = comp_fn(*[out_tensors[idx] for idx in args])

                # print(out)
                # return (out, )
                ## SPLIT PIPELINES
                print("PRINTING FINAL GM GRAPH" , gm.graph)


                return gm.forward(*args)

                # output_node = gm.graph.find_nodes(op="output")[0].args[0][0]

                # nonlocal out_size
                # out_size = output_node.meta['example_value'].size()

                # out_tensor = torch.empty(list(map(int, out_size)))

                # # out_tensor = torch.empty(size=output_node.meta['example_value'].size())

                # l_x = Variable.init("l_x")
                # l_y = Variable.init("l_y")

                # nonlocal output_adt_ptr
                # output_adt_ptr = env[output_node.name]

                # program = Composable([
                #     Tile(output_adt_ptr["row"], l_x)(
                #         Tile(output_adt_ptr["col"], l_y)(
                #             *fn_calls
                #         )
                #     )
                # ])

                # generated_runner = Runner(program)

                # generated_runner.compile(cpuRunner(["matrix"]))

                # l_x_val = Int.init(tile_rows)
                # l_y_val = Int.init(out_size[1])


                # variables.append((l_x, l_x_val))
                # variables.append((l_y, l_y_val))

                # nonlocal input_adt_ptr_to_arg_idx
                # input_adt_ptr_to_arg_idx = {
                #     env[input_arg_name]: input_arg_idx
                #     for input_arg_name, input_arg_idx in input_name_to_arg_idx.items()
                # }

                # gern_args = {
                #     var.getName(): getAddress(var_val)
                #     for var, var_val in variables
                # } | {
                #     input_adt_ptr.getName(): getAddress(
                #         MatrixCPU.init(args[i].data_ptr(), *args[i].size(), args[i].stride()[0])
                #     )
                #     for input_adt_ptr, i in input_adt_ptr_to_arg_idx.items()
                # } | {
                #     output_adt_ptr.getName(): getAddress(
                #         MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                #     )
                # }
                # # print("RUNNER EVALUATE")
                # generated_runner.evaluate(gern_args)

                # return (out_tensor,)
                # return gm.forward(*args)
            else:
                print("REUSING GENERATED RUNNER")

                # out_tensor = torch.empty(list(map(int, out_size)))

                # gern_args = {
                #     var.getName(): getAddress(var_val)
                #     for var, var_val in variables
                # } | {
                #     input_adt_ptr.getName(): getAddress(
                #         MatrixCPU.init(args[i].data_ptr(), *args[i].size(), args[i].stride()[0])
                #     )
                #     for input_adt_ptr, i in input_adt_ptr_to_arg_idx.items()
                # } | {
                #     output_adt_ptr.getName(): getAddress(
                #         MatrixCPU.init(out_tensor.data_ptr(), *out_tensor.size(), out_tensor.stride()[0])
                #     )
                # }

                # generated_runner.evaluate(gern_args)

                ## SPLIT PIPELINE
                # out = None

                # out_tensors = []
                # for generated_runner, gern_args, out_tensor in pipeline_sections:
                #     generated_runner.evaluate(gern_args)
                #     out_tensors.append(out_tensor)

                # for comp_fn, args in python_computations:
                #     out = comp_fn(*[out_tensors[idx] for idx in args])

                # return (out,)
                ## SPLIT PIPELINE

                return gm.forward(*args)
        return custom_callable
    
    # opt_M = torch.compile(M(), backend=custom_backend, dynamic=True)
    opt_M = torch.compile(M, backend=custom_backend)
    return opt_M
   
    