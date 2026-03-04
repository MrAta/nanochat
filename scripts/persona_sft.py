"""
Persona SFT: fine-tune an existing SFT checkpoint to learn a persona from
Telegram-derived conversation data.

Key differences from chat_sft.py:
  - Loads from "sft" checkpoint (chatsft_checkpoints/) instead of "base"
  - Saves to persona_checkpoints/ instead of chatsft_checkpoints/
  - Oversample Telegram data via --telegram-weight (default 10x)
  - Optionally mix in a diluted SmolTalk pass (--keep-general) to reduce
    catastrophic forgetting
  - Lower default --init-lr-frac (0.3) to preserve learned representations

Run as:
    torchrun --standalone --nproc_per_node=1 -m scripts.persona_sft -- \\
        --telegram-data data/telegram_sft.jsonl \\
        --telegram-weight 10 \\
        --init-lr-frac 0.3 \\
        --run persona_run1
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from contextlib import nullcontext
from nanochat.common import (compute_init, compute_cleanup, print0,
                              DummyWandb, get_base_dir, autodetect_device_type,
                              get_peak_flops)
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine

from tasks.common import TaskMixture
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Persona SFT from an SFT checkpoint")
# Logging
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
# Model loading — from SFT checkpoint
parser.add_argument("--model-tag", type=str, default=None,
                    help="SFT model tag to fine-tune from")
parser.add_argument("--model-step", type=int, default=None,
                    help="SFT step to load from (default: latest)")
parser.add_argument("--load-optimizer", type=int, default=0,
                    help="Warm-start optimizer from SFT checkpoint (0=no, 1=yes; default 0)")
# Training data
parser.add_argument("--telegram-data", type=str, default="data/telegram_sft.jsonl",
                    help="Path to Telegram SFT JSONL file")
parser.add_argument("--telegram-weight", type=int, default=10,
                    help="Repeat Telegram data N times (oversampling; default: 10)")
parser.add_argument("--keep-general", action="store_true", default=False,
                    help="Include one epoch of SmolTalk to reduce catastrophic forgetting")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1,
                    help="number of optimization steps (-1 = full epoch)")
# Batch sizes (inherit from SFT checkpoint by default)
parser.add_argument("--max-seq-len", type=int, default=None)
parser.add_argument("--device-batch-size", type=int, default=None)
parser.add_argument("--total-batch-size", type=int, default=None)
# Optimization
parser.add_argument("--embedding-lr", type=float, default=None)
parser.add_argument("--unembedding-lr", type=float, default=None)
parser.add_argument("--matrix-lr", type=float, default=None)
parser.add_argument("--init-lr-frac", type=float, default=0.3,
                    help="Initial LR as fraction of SFT checkpoint LR (default: 0.3)")
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
# Evaluation
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--eval-tokens", type=int, default=10*524288)
# Output
parser.add_argument("--output-dir", type=str, default=None,
                    help="Override persona checkpoint directory (default: persona_checkpoints/<tag>)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                if device_type == "cuda" else nullcontext())
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (DummyWandb() if use_dummy_wandb
             else wandb.init(project="nanochat-persona-sft", name=args.run,
                             config=user_config))

if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback.")

# Load from SFT checkpoint (not base)
model, tokenizer, meta = load_model("sft", device, phase="train",
                                     model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from SFT checkpoint
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,   meta),
    ("device_batch_size", 32,     meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,    pretrain_user_config),
    ("unembedding_lr",    0.004,  pretrain_user_config),
    ("matrix_lr",         0.02,   pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from SFT checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides SFT value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Build training mixture
base_dir = get_base_dir()
telegram_path = os.path.join(base_dir, args.telegram_data) if not os.path.isabs(args.telegram_data) else args.telegram_data
telegram_task = CustomJSON(filepath=telegram_path)
print0(f"Telegram data: {len(telegram_task)} rows (x{args.telegram_weight} oversampling)")

train_tasks = [telegram_task] * args.telegram_weight

if args.keep_general:
    train_tasks.append(SmolTalk(split="train"))
    print0("Including SmolTalk mix (1 epoch) to reduce catastrophic forgetting")

train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows total")

val_dataset = TaskMixture([SmolTalk(split="test")])

# Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0,
)

if args.load_optimizer:
    optimizer_data = load_optimizer_state("sft", device, rank=ddp_rank,
                                          model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from SFT checkpoint")
    else:
        print0("WARNING: SFT optimizer checkpoint not found, starting fresh")

# Apply init_lr_frac
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Data loader (same BOS-aligned bestfit packing as chat_sft.py)
last_step = False
approx_progress = 0.0
current_epoch = 1

def sft_data_generator_bos_bestfit(split, buffer_size=100):
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            content_len = 0
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1
        for i, cl in enumerate(row_lengths):
            if cl < row_capacity:
                targets[i, cl-1:] = -1

        yield inputs, targets


train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0.0


def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac


def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# Determine output directory
if args.output_dir:
    checkpoint_dir = args.output_dir
else:
    tag = args.model_tag if args.model_tag else f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "persona_checkpoints", tag)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
x, y = next(train_loader)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    if last_step:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
            },
            rank=ddp_rank,
        )
        print0(f"Saved persona checkpoint to: {checkpoint_dir}")
        break

    # Single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
        progress = max(progress, approx_progress)

    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    step += 1
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | "
           f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
           f"mfu: {mfu:.2f} | epoch: {current_epoch}")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

wandb_run.finish()
compute_cleanup()
