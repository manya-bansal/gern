import torch
from transformers import BertModel
from generate_torch_compile_pipeline import gen, FnInterface
from gern_py import *
import operator

torch_to_gern = {
    torch.t: FnInterface(MatrixTranspose),
    operator.matmul: FnInterface(MatrixMultiply, {"shared_len": lambda args: args[0].meta["example_value"].size(dim=1)}),
    operator.truediv: FnInterface(MatrixDivn),
    torch.nn.functional.softmax: FnInterface(MatrixSoftmax)
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
    return gm.forward

# Load a pre-trained model
model = BertModel.from_pretrained("bert-base-uncased")

# Move model to CPU
model.to("cpu")

# Generate a sample input (batch_size=1, seq_length=128)
input_ids = torch.randint(0, 30522, (1, 128), dtype=torch.long)
attention_mask = torch.ones((1, 512), dtype=torch.long)

# opt_model = torch.compile(model, backend=custom_backend)
output = model(input_ids, attention_mask=attention_mask)
print(output["pooler_output"])

gern_model = gen(model, torch_to_gern, tile_rows=128)
print(gern_model(input_ids))