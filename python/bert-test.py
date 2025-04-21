import torch
from transformers import BertModel
from generate_torch_compile_pipeline_4d import gen as gen4d, FnInterface
from generate_torch_compile_pipeline import gen as gen2d
from gern_py import *
import operator
import torch.utils.benchmark as benchmark

torch_to_gern_4d = {
    torch.nn.functional.scaled_dot_product_attention: FnInterface(MatrixAttention4D, extra_args={"height": lambda args: args[0].meta["example_value"].size(dim=-2), "width": lambda args: args[0].meta["example_value"].size(dim=-1)}),
    "transpose": FnInterface(MatrixTranspose4D, fn_args=(lambda args: args[1], lambda args: args[2]), skip_nontensor_args=True)
}

torch_to_gern_2d = {
    torch.t: FnInterface(MatrixTranspose),
    operator.matmul: FnInterface(MatrixMultiply, extra_args={"shared_len": lambda args: args[0].meta["example_value"].size(dim=-1)}),
    operator.truediv: FnInterface(MatrixDivn),
    torch.nn.functional.softmax: FnInterface(MatrixSoftmax),
    torch.nn.functional.scaled_dot_product_attention: FnInterface(MatrixAttention, extra_args={"height": lambda args: args[0].meta["example_value"].size(dim=-2), "width": lambda args: args[0].meta["example_value"].size(dim=-1)})
}


def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        print("NODE", node)
        if node.target == torch.nn.functional.scaled_dot_product_attention:
            for arg in node.args:
                print("arg", arg, arg.meta)
            for arg in node.kwargs:
                print("kwarg", arg)
    return gm.forward

# Load a pre-trained model
model = BertModel.from_pretrained("bert-base-uncased")
model2 = BertModel.from_pretrained("bert-base-uncased")

# Move model to CPU
model.to("cpu")
model2.to("cpu")

# Generate a sample input (batch_size=1, seq_length=512)
input_ids = torch.randint(0, 30522, (1, 512), dtype=torch.long)
attention_mask = torch.ones((1, 512), dtype=torch.long)

# opt_model = torch.compile(model, backend=custom_backend)
# print(opt_model(input_ids, attention_mask=attention_mask))

output = model2(input_ids)
ref_output = output["pooler_output"]
print("REF OUTPUT", ref_output)

# gern_model_2d = gen2d(model, torch_to_gern_2d, tile_rows=512)
# gern_output_2d = gern_model_2d(input_ids)["pooler_output"]
# # print("GERN OUTPUT", gern_output)

gern_model_4d = gen4d(model, torch_to_gern_4d, tile_rows=512)
gern_output_4d = gern_model_4d(input_ids)["pooler_output"]

# print(torch.allclose(ref_output, gern_output_2d, atol=1e-5))
print(torch.allclose(ref_output, gern_output_4d, atol=1e-5))
