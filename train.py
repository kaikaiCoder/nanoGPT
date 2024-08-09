from dataclasses import dataclass
import torch
import math
import time
import torch.nn.functional as F
import os
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from hella_swag import render_example, iterate_examples
from contextlib import nullcontext
from model import GPTConfig, GPT
from data_loader import DataLoader

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
backend = "nccl"
total_batch_size = 2**19 # 2**19 = 524288
gradient_accumulation_steps = 5 * 8  # accumulate gradients over N steps
batch_size = 16
block_size = 1024
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
max_steps = 19073
useCompile = False
eval_interval = 1000  # evaluate the model every N steps
sample_invterval = 2500  # sample the model every N steps

out_dir = "out"
log_dir = "log"

# ======================= train =======================

# set up distributed data parallel
# DDP launch  for 2 GPUs:
# torchrun --standalone --nproc_per_node=2 train_gpt.py
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    print("using distributed data parallel")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (
        ddp_rank == 0
    )  # only the master process will print, write to disk, etc.
    seed_offset = ddp_rank  # each process gets a different random seed
    assert (
        gradient_accumulation_steps % ddp_world_size == 0
    )  # make sure gradient accumulation is divisible by world size
    gradient_accumulation_steps //= (
        ddp_world_size  # each process will accumulate gradients over fewer steps
    )
else:
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if device.startswith("cuda") else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
if device_type == "cuda":
    torch.set_float32_matmul_precision("high")
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

assert (
    total_batch_size % (batch_size * block_size * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B*T*ddp_world_size"
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(
        f"=> gradient accumulation steps: {total_batch_size} % {batch_size} * {block_size} * {ddp_world_size} = {gradient_accumulation_steps}"
    )

train_loader = DataLoader(
    batch_size,
    block_size,
    process_rank=ddp_rank,
    num_process=ddp_world_size,
    split="train",
)
val_loader = DataLoader(
    batch_size,
    block_size,
    process_rank=ddp_rank,
    num_process=ddp_world_size,
    split="val",
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = GPT(GPTConfig())
elif init_from == "resume":
    # find the latest checkpoint file
    checkpoint_files = [
        f for f in os.listdir(out_dir) if f.startswith("model_") and f.endswith(".pt")
    ]
    assert len(checkpoint_files) > 0, "no model checkpoints found"
    checkpoint_files = sorted(checkpoint_files)
    checkpoint_file = checkpoint_files[-1]
    checkpoint_path = os.path.join(out_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    model = GPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    step = checkpoint["step"]
    print(f"resuming from step {step}")
elif init_from.startswith("gpt2"):
    print(f"Initializing from pretrained GPT2 model {init_from}")
    model = GPT.from_pretrained(init_from)

# model = GPT(GPTConfig())
model.to(device)
if useCompile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


def get_lr(it):
    max_lr = 6e-4
    min_lr = 0.1 * max_lr
    warmup_steps = 715
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])


@torch.no_grad()
def val():
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    val_loss_steps = 20
    for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()

    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        if step > 0 and (step % 5000 == 0 or step == max_steps - 1):
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_path = os.path.join(out_dir, f"model_{step:05d}.pt")
            checkpoint = {
                "model": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
                "val_loss": val_loss_accum.item(),
            }
            torch.save(checkpoint, checkpoint_path)


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


@torch.no_grad()
def sample():
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with ctx:
            logits, loss = model(tokens)  # (4 * seq_len_vocab)
        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(
            num_correct_norm, dtype=torch.long, device=device
        )
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")


os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
step = 0
while True:
    last_step = step == max_steps - 1
    t0 = time.time()
    if not useCompile and (step % eval_interval == 0 or last_step):
        val()
    if (step % sample_invterval == 0 or last_step) and (not useCompile):
        sample()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    token_processed = (
        train_loader.B * train_loader.T * gradient_accumulation_steps * ddp_world_size
    )
    if master_process:
        print(
            f"step {step:4d} | loss: {loss_accum.item():.6f} | lr:{lr:.4e} | dt: {dt*1000:.2f} | norm: {norm:.4f} | tok/sec: {token_processed/dt:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
    step += 1
    if step > max_steps:
        break
if ddp:
    destroy_process_group()
