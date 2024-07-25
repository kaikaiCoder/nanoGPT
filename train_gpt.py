from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import time
import torch.nn.functional as F
import inspect
import os
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
import numpy as np
from hella_swag import render_example, iterate_examples


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader:

    def __init__(self, B, T, process_rank, num_process, split):
        self.B = B
        self.T = T
        self.procee_rank = process_rank
        self.num_process = num_process
        assert split in {"train", "val"}

        data_dir = "edu_fineweb10B"
        shard = os.listdir(data_dir)
        shard = [s for s in shard if split in s]
        shard = sorted(shard)
        self.shard = shard
        assert len(shard) > 0, f"no data files found in {data_dir}"
        if master_process:
            print(f"found {len(shard)} shards for {split}")
        self.current_shard = 0
        self.tokens = load_tokens(self.shard[self.current_shard])

        # with open("input.txt", "r") as f:
        #     text = f.read()
        #     text = text
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(tokens)} tokens")
        self.current_position = B * T * process_rank
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.procee_rank
        self.current_shard = 0
        self.tokens = load_tokens(self.shard[self.current_shard])

    def next_batch(self):
        B, T, start = self.B, self.T, self.current_position
        end = self.current_position + B * T + 1
        buf = self.tokens[start:end]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_process
        if self.current_position + B * T * self.num_process + 1 > len(self.tokens):
            # if we reach the end of the shard, move to the next shard
            self.current_shard = (self.current_shard + 1) % len(self.shard)
            self.tokens = load_tokens(self.shard[self.current_shard])
            self.current_position = B * T * self.procee_rank
        return x, y


@dataclass
class GPTConfig:

    # GPT2-124B https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/gpt2#openai-gpt2
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # mask to prevent attention to future tokens max_sequence_length x max_sequence_length
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding size
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)  # B, nh, T, hs
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init_weight__)

    def __init_weight__(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in non_decay_params)
        print(
            f"number decay of parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
        )
        print(
            f"number non-decay of parameter tensors: {len(num_decay_params)}, with {num_non_decay_params} parameters"
        )

        # create AdamW optimizer and use the fused version if possible
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def forward(self, x, y):
        B, T = x.size()
        assert (
            T <= self.config.block_size
        ), "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(x)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # B, T, V
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        # load GPT2 model weight2 from huggingface
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config = GPTConfig(**config_args[model_type])
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


# ======================= train =======================
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# set up distributed data parallel
# DDP launch  for 2 GPUs:
# torchrun --standalone --nproc_per_node=2 train_gpt.py
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    print("using distributed data parallel")
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print("using device:", device)

# gradient accumulation
total_batch_size = 2**19  # ~0.5M tokens
B = 64
T = 1024
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_acc_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> gradient accumulation steps: {grad_acc_steps}")

train_loader = DataLoader(
    B, T, process_rank=ddp_rank, num_process=ddp_world_size, split="train"
)
val_loader = DataLoader(
    B, T, process_rank=ddp_rank, num_process=ddp_world_size, split="val"
)

if device == "cuda":
    torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig())
model.to(device)
useCompile = False
if useCompile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.moudle if ddp else model

max_lr = 3e-4
min_lr = 0.1 * max_lr
warmup_steps = 715
max_steps = 19073 * 4


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(0.1, 3e-4, device)


def val(step,last_step):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if i > 0 and (i % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{i:05d}.pt")
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


def sample(step):
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
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
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


log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

for i in range(max_steps):
    last_step = i == max_steps - 1
    t0 = time.time()
    if not useCompile and (i % 100 == 0 or last_step):
        val(i,last_step)
    if ((i > 0 and i % 100 == 0) or last_step) and (not useCompile):
        sample(i)
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_acc_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_acc_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    token_processed = train_loader.B * train_loader.T * grad_acc_steps * ddp_world_size
    if master_process:
        print(
            f"step {i:4d} | loss: {loss_accum.item():.6f} | lr:{lr:.4e} | dt: {dt*1000:.2f} | norm: {norm:.4f} | tok/sec: {token_processed/dt:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{i} train  {loss_accum.itme():.4f}")
if ddp:
    destroy_process_group()
