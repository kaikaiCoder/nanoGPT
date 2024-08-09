import os
import torch
import tiktoken
from contextlib import nullcontext

from model import GPT

init_from = "resume"
out_dir = "out"

start = "\n"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)

device = "cuda"
device_type = "cuda" if "cuda" in device else "cpu"
compile = False
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


if init_from == "resume":
    checkpoint_path = os.path.join(out_dir, "model_15000.pt")
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]
    model = GPT(config)
    stat_dict = checkpoint["model"]
    model.load_state_dict(stat_dict)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from)

model.eval()
model.to(device_type)

if compile:
    model = torch.compile(model)

enc = tiktoken.get_encoding("gpt2")

if start.startswith("FILE:"):
    with open(start[5:], "r") as f:
        start = f.read()
ids = enc.encode(start, allowed_special={"<|endoftext|>"})
x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature, top_k)
            print(enc.decode(y[0].tolist()))
            print("---------------")
