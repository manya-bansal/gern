import torch
from transformers import BertModel
from generate_torch_compile_pipeline import gen, FnInterface
from gern_py import *
import operator
import torch.utils.benchmark as benchmark

torch_to_gern = {
    torch.t: FnInterface(MatrixTranspose),
    operator.matmul: FnInterface(MatrixMultiply, {"shared_len": lambda args: args[0].meta["example_value"].size(dim=-1)}),
    operator.truediv: FnInterface(MatrixDivn),
    torch.nn.functional.softmax: FnInterface(MatrixSoftmax),
    torch.nn.functional.scaled_dot_product_attention: FnInterface(MatrixAttention, {"height": lambda args: args[0].meta["example_value"].size(dim=-2), "width": lambda args: args[0].meta["example_value"].size(dim=-1)})
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

# Move model to CPU
model.to("cpu")

# Generate a sample input (batch_size=1, seq_length=512)
input_ids = torch.randint(0, 30522, (1, 512), dtype=torch.long)
attention_mask = torch.ones((1, 512), dtype=torch.long)

# opt_model = torch.compile(model, backend=custom_backend)
# print(opt_model(input_ids, attention_mask=attention_mask))

output = model(input_ids)
ref_output = output["pooler_output"]
# print("REF OUTPUT", ref_output)

gern_model = gen(model, torch_to_gern, tile_rows=512)
gern_output = gern_model(input_ids)["pooler_output"]
# print("GERN OUTPUT", gern_output)

print(torch.allclose(ref_output, gern_output, atol=1e-5))

results = []
optimized = benchmark.Timer(
    setup='gern_model(input_ids)',
    stmt='gern_model(input_ids)',
    globals={'gern_model': gern_model, 'input_ids': input_ids},
    label=f"bert batch_size=1, seq_length=512",
    description=f"gern",
    sub_label=f"gern"
)
results.append(optimized.blocked_autorange(min_run_time=2))

unoptimized = benchmark.Timer(
    setup='model(input_ids)',
    stmt='model(input_ids)',
    globals={'model': model, 'input_ids': input_ids},
    label=f"bert batch_size=1, seq_length=512",
    description=f"default",
    sub_label=f"default"
)
results.append(unoptimized.blocked_autorange(min_run_time=2))

compare = benchmark.Compare(results)
compare.print()

