#!/usr/bin/env python3
"""
train_baseline.py — Fine-tune Qwen3.5-VL on the default multi-dataset mix
(QVID + Kinetics-400 + HMDB51 + ShareGPT4Video) WITHOUT pose features.
Same data, same epochs, same hyperparameters as train.py, but no projector
and no pose token injection.  This provides a fair finetuning baseline.

Usage
-----
  accelerate launch train_baseline.py \
      --model_name Qwen/Qwen3.5-VL-7B-Instruct \
      --output_dir /orcd/compute/ppliang/001/qwen_multi_baseline
"""

import argparse
import functools
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# Workaround: PreTrainedModel.dtype iterates self.parameters() and raises
# StopIteration when called on an FSDP-wrapped submodule whose params are
# momentarily empty/flat. Patch it to fall back to bf16 in that case.
_orig_pretrained_dtype = PreTrainedModel.dtype.fget

def _safe_pretrained_dtype(self):
    try:
        return _orig_pretrained_dtype(self)
    except StopIteration:
        return torch.bfloat16

PreTrainedModel.dtype = property(_safe_pretrained_dtype)

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from multi_dataset import get_all_samples
from train import MultiVideoDataset, _sample_frames, _load_pose_features


# ── Answer matching (mirrors train.py) ─────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

_STOP_WORDS = {"a", "an", "the", "my", "your", "his", "her", "its", "our", "their"}

def _normalize_lenient(text: str):
    text = _normalize(text)
    words = [w for w in text.split() if w not in _STOP_WORDS]
    text = " ".join(words) if words else text
    return text, text.replace(" ", "")

def _extract_mc_letter(text: str) -> str:
    m = re.search(r"\b([A-Ea-e])\b", text)
    if m:
        return m.group(1).upper()
    c = text.strip()[:1].upper()
    return c if c in "ABCDE" else ""

def _is_correct(pred: str, gt: str, task_type: str) -> bool:
    if task_type == "multiple_choice":
        return _extract_mc_letter(pred) == gt.strip().upper()
    return _normalize(pred) == _normalize(gt)

def _is_correct_lenient(pred: str, gt: str, task_type: str) -> bool:
    if task_type == "multiple_choice":
        return _extract_mc_letter(pred) == gt.strip().upper()
    np_, ng = _normalize(pred), _normalize(gt)
    if np_ == ng:
        return True
    lp, lp_ns = _normalize_lenient(pred)
    lg, lg_ns = _normalize_lenient(gt)
    if lp == lg or lp_ns == lg_ns:
        return True
    if lp in lg or lg in lp:
        return True
    if set(lp.split()) == set(lg.split()):
        return True
    return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Collate (no pose tokens) ────────────────────────────────────────────────

def _collate(batch: List[dict], processor, tokenizer) -> dict:
    """
    Build processor inputs for a batch.  Labels are -100 for prompt positions;
    only answer tokens have real labels.
    """
    full_texts, prompt_texts, all_images = [], [], []

    for ex in batch:
        content = [{"type": "image"} for _ in ex["pil_images"]]
        content.append({"type": "text", "text": ex["question"]})
        user_msg = {"role": "user", "content": content}
        asst_msg = {"role": "assistant", "content": ex["answer"]}

        full_texts.append(
            processor.apply_chat_template(
                [user_msg, asst_msg], tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        prompt_texts.append(
            processor.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        all_images.append(ex["pil_images"])

    # Compute answer token lengths (text-only; no visual tokens in answer)
    answer_lens = []
    for full_t, prompt_t in zip(full_texts, prompt_texts):
        answer_text = full_t[len(prompt_t):]
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        answer_lens.append(len(answer_ids))

    # Full tokenization with visual tokens
    inputs = processor(
        text=full_texts, images=all_images, return_tensors="pt", padding=True,
    )
    input_ids = inputs["input_ids"]  # (B, seq_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, seq_len = input_ids.shape
    labels = torch.full((B, seq_len), -100, dtype=input_ids.dtype)

    for i, ans_len in enumerate(answer_lens):
        nonpad = (input_ids[i] != pad_id).sum().item()
        ans_start = max(1, nonpad - ans_len)
        # Label only the answer tokens
        labels[i, ans_start:nonpad] = input_ids[i, ans_start:nonpad]

    result = {
        "input_ids":      input_ids,
        "attention_mask":  inputs["attention_mask"],
        "labels":          labels,
    }
    if "mm_token_type_ids" in inputs:
        result["mm_token_type_ids"] = inputs["mm_token_type_ids"]
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in inputs:
            result[k] = inputs[k]
    return result


def _collate_gen(batch: List[dict], processor, tokenizer) -> dict:
    """Build prompt-only inputs for generative validation."""
    texts, all_images = [], []
    for ex in batch:
        content = [{"type": "image"} for _ in ex["pil_images"]]
        content.append({"type": "text", "text": ex["question"]})
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )
        all_images.append(ex["pil_images"])

    inputs = processor(text=texts, images=all_images, return_tensors="pt", padding=True)
    return {
        **{k: v for k, v in inputs.items()},
        "answers":    [ex["answer"]    for ex in batch],
        "datasets":   [ex["dataset"]   for ex in batch],
        "task_types": [ex["task_type"] for ex in batch],
    }


# ── Generative validation ──────────────────────────────────────────────────

def _stratified_subset(dataset, max_samples: int):
    """Return a Subset that samples proportionally from each dataset."""
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    by_ds = defaultdict(list)
    for idx in range(len(dataset)):
        ex = dataset[idx] if not isinstance(dataset, torch.utils.data.Subset) else dataset.dataset[dataset.indices[idx]]
        by_ds[ex["dataset"]].append(idx)

    total = sum(len(v) for v in by_ds.values())
    selected = []
    for ds_name, indices in sorted(by_ds.items()):
        n = max(1, round(max_samples * len(indices) / total))
        step = max(1, len(indices) // n)
        selected.extend(indices[::step][:n])

    selected = selected[:max_samples]
    return torch.utils.data.Subset(dataset, selected)


def _run_generative_validation(
    model, val_ds, processor, tokenizer, accelerator,
    max_new_tokens: int = 32, max_samples: int = 0,
) -> dict:
    """Generate and compute strict/lenient accuracy."""
    is_main = accelerator.is_main_process

    subset = _stratified_subset(val_ds, max_samples)

    collate_fn = lambda b: _collate_gen(b, processor, tokenizer)
    gen_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
    gen_loader = accelerator.prepare(gen_loader)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    overall = {"correct": 0, "total": 0}
    overall_lenient = {"correct": 0, "total": 0}
    by_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    by_dataset_len = defaultdict(lambda: {"correct": 0, "total": 0})
    by_tasktype = defaultdict(lambda: {"correct": 0, "total": 0})
    by_tasktype_len = defaultdict(lambda: {"correct": 0, "total": 0})

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in tqdm(gen_loader, desc="Gen-val", leave=False, disable=not is_main):
            answers    = batch.pop("answers")
            datasets   = batch.pop("datasets")
            task_types = batch.pop("task_types")

            prompt_len = batch["input_ids"].shape[1]
            gen_kwargs = {k: v for k, v in batch.items() if v is not None}
            try:
                gen_ids = model.generate(
                    **gen_kwargs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                new_ids = gen_ids[:, prompt_len:]
            except Exception as e:
                log.warning("Gen-val generation failed, skipping: %s", e)
                continue

            preds = [tokenizer.decode(ids, skip_special_tokens=True).strip()
                     for ids in new_ids]

            for pred, gt, ds, tt in zip(preds, answers, datasets, task_types):
                c  = _is_correct(pred, gt, tt)
                cl = _is_correct_lenient(pred, gt, tt)
                overall["correct"]             += int(c)
                overall["total"]               += 1
                overall_lenient["correct"]     += int(cl)
                overall_lenient["total"]       += 1
                by_dataset[ds]["correct"]      += int(c)
                by_dataset[ds]["total"]        += 1
                by_dataset_len[ds]["correct"]  += int(cl)
                by_dataset_len[ds]["total"]    += 1
                by_tasktype[tt]["correct"]     += int(c)
                by_tasktype[tt]["total"]       += 1
                by_tasktype_len[tt]["correct"] += int(cl)
                by_tasktype_len[tt]["total"]   += 1

    if was_training:
        model.train()

    def acc(s):
        return s["correct"] / max(1, s["total"])

    return {
        "overall_strict":  acc(overall),
        "overall_lenient": acc(overall_lenient),
        "total": overall["total"],
        "correct_strict": overall["correct"],
        "correct_lenient": overall_lenient["correct"],
        "by_dataset": {
            ds: {"strict": acc(s), "lenient": acc(by_dataset_len[ds]),
                 "correct": s["correct"], "total": s["total"]}
            for ds, s in sorted(by_dataset.items())
        },
        "by_task_type": {
            tt: {"strict": acc(s), "lenient": acc(by_tasktype_len[tt]),
                 "correct": s["correct"], "total": s["total"]}
            for tt, s in sorted(by_tasktype.items())
        },
    }


def _log_gen_results(label: str, results: dict, log_fn=log.info):
    log_fn(
        "%s: strict=%.4f  lenient=%.4f  (%d/%d/%d)",
        label, results["overall_strict"], results["overall_lenient"],
        results["correct_strict"], results["correct_lenient"], results["total"],
    )
    for ds, s in results["by_dataset"].items():
        log_fn("    %s: strict=%.4f  lenient=%.4f  (%d/%d)",
               ds, s["strict"], s["lenient"], s["correct"], s["total"])
    for tt, s in results.get("by_task_type", {}).items():
        log_fn("    %s: strict=%.4f  lenient=%.4f  (%d/%d)",
               tt, s["strict"], s["lenient"], s["correct"], s["total"])


# ── Training ────────────────────────────────────────────────────────────────

def train(args):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        auto_wrap_policy=functools.partial(
            size_based_auto_wrap_policy, min_num_params=int(1e8)
        ),
        use_orig_params=True,
        cpu_offload=False,
        activation_checkpointing=False,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        fsdp_plugin=fsdp_plugin,
    )
    set_seed(args.seed)
    device = accelerator.device
    is_main = accelerator.is_main_process

    # ── Log file (main process only) ─────────────────────────────────────
    if is_main:
        from datetime import datetime
        log_dir = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)
        log.info("Logging to %s", log_file)

    log.info("Accelerator: num_processes=%d  device=%s", accelerator.num_processes, device)

    # ── WandB (main process only) ────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and not args.no_wandb and is_main
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        log.info("WandB run: %s", wandb.run.name)
    elif is_main and not args.no_wandb and not WANDB_AVAILABLE:
        log.warning("wandb not installed — logging to console only.")

    # ── Resume checkpoint detection ──────────────────────────────────────
    resume_ckpt = Path(args.resume_ckpt)
    resume_state = None
    if (resume_ckpt / "train_state.json").exists():
        with open(resume_ckpt / "train_state.json") as f:
            resume_state = json.load(f)
        if is_main:
            log.info("Found resume checkpoint at %s (next_epoch=%d)",
                     resume_ckpt, resume_state["next_epoch"])

    # ── Model ────────────────────────────────────────────────────────────
    log.info("Loading model: %s", args.model_name)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    for p in model.parameters():
        p.requires_grad = True
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled.")

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Total params: %s", f"{total_params:,}")

    # ── Datasets ─────────────────────────────────────────────────────────
    splits = get_all_samples()
    train_ds = MultiVideoDataset(
        splits["train"], fps=args.fps, max_frames=args.max_frames,
        pose_sample_n=args.pose_sample_n,
    )
    val_ds = MultiVideoDataset(
        splits["val"], fps=args.fps, max_frames=args.max_frames,
        pose_sample_n=args.pose_sample_n,
    )

    collate_fn = lambda b: _collate(b, processor, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    # ── Prepare (FSDP wrap) ──────────────────────────────────────────────
    model, train_loader, val_loader = accelerator.prepare(
        model, train_loader, val_loader
    )

    if args.torch_compile:
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )

    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.num_epochs
    warmup_steps = round(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    # ── Training loop ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1

    resume_step_in_epoch = 0
    if resume_state:
        start_epoch   = resume_state["next_epoch"]
        global_step   = resume_state["global_step"]
        best_val_loss = resume_state["best_val_loss"]
        resume_step_in_epoch = resume_state.get("step_in_epoch", 0)
        accelerator.load_state(str(resume_ckpt / "accel_state"))
        if is_main:
            log.info(
                "Resumed: next_epoch=%d  global_step=%d  step_in_epoch=%d  best_val_loss=%.4f",
                start_epoch, global_step, resume_step_in_epoch, best_val_loss,
            )

    optimizer.zero_grad()

    # ── Warm up FSDP ─────────────────────────────────────────────────────
    if is_main:
        log.info("Initializing FSDP …")
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        model(input_ids=_dummy, attention_mask=torch.ones_like(_dummy))
    accelerator.wait_for_everyone()

    # ── Training epochs ──────────────────────────────────────────────────
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        running_loss = 0.0

        if resume_step_in_epoch > 0:
            active_loader = accelerator.skip_first_batches(
                train_loader, resume_step_in_epoch
            )
            step_offset = resume_step_in_epoch
            if is_main:
                log.info("Skipping first %d batches of epoch %d",
                         resume_step_in_epoch, epoch)
            resume_step_in_epoch = 0
        else:
            active_loader = train_loader
            step_offset = 0

        pbar = tqdm(
            active_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
            leave=True, disable=not is_main,
            initial=step_offset, total=len(train_loader),
        )

        for local_step, batch in enumerate(pbar):
            step = local_step + step_offset
            with accelerator.accumulate(model):
                labels = batch.pop("labels")

                outputs = model(**batch, labels=labels)
                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if accelerator.sync_gradients:
                global_step += 1

                step_metrics = {"epoch": epoch} if use_wandb else {}

                if global_step % args.save_steps == 0:
                    torch.cuda.synchronize()
                    _save_resume_checkpoint(
                        accelerator, resume_ckpt,
                        next_epoch=epoch, global_step=global_step,
                        step_in_epoch=step + 1,
                        best_val_loss=best_val_loss,
                    )

                    if args.val_steps > 0 and global_step % args.val_steps == 0:
                        if is_main:
                            log.info("Running generative validation at step %d …", global_step)
                        gen_results = _run_generative_validation(
                            model, val_ds, processor, tokenizer, accelerator,
                            max_new_tokens=args.val_max_new_tokens,
                            max_samples=args.val_max_samples,
                        )
                        if is_main:
                            _log_gen_results(f"  gen-val step {global_step}", gen_results)
                            step_metrics.update({
                                "val_gen/accuracy_strict":  gen_results["overall_strict"],
                                "val_gen/accuracy_lenient": gen_results["overall_lenient"],
                                **{f"val_gen/{ds}_strict": s["strict"]
                                   for ds, s in gen_results["by_dataset"].items()},
                                **{f"val_gen/{ds}_lenient": s["lenient"]
                                   for ds, s in gen_results["by_dataset"].items()},
                            })

                if global_step % args.log_steps == 0 and is_main:
                    n = args.log_steps * args.grad_accum
                    avg_loss = running_loss / n
                    running_loss = 0.0

                    cur_lr = scheduler.get_last_lr()[0]

                    log.info(
                        "epoch %d  step %d  loss %.4f  lr %.2e",
                        epoch, global_step, avg_loss, cur_lr,
                    )
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    step_metrics.update({
                        "train/loss": avg_loss,
                        "train/lr":   cur_lr,
                    })

                if use_wandb and step_metrics and is_main:
                    wandb.log(step_metrics, step=global_step)

        # ── Validation (loss) ────────────────────────────────────────────
        model.eval()
        val_loss_sum = torch.zeros(1, device=device)
        n_batches    = torch.zeros(1, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False,
                              disable=not is_main):
                labels = batch.pop("labels")
                outputs = model(**batch, labels=labels)
                val_loss_sum += outputs.loss.detach()
                n_batches    += 1

        val_loss_sum = accelerator.reduce(val_loss_sum, reduction="sum")
        n_batches    = accelerator.reduce(n_batches,    reduction="sum")
        val_loss = (val_loss_sum / n_batches.clamp(min=1)).item()

        if is_main:
            log.info("=== Epoch %d  val_loss=%.4f ===", epoch, val_loss)
            if use_wandb:
                wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        # ── Generative validation (full val set, end of epoch) ───────────
        if is_main:
            log.info("Running end-of-epoch generative validation …")
        gen_results = _run_generative_validation(
            model, val_ds, processor, tokenizer, accelerator,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=0,
        )
        if is_main:
            _log_gen_results(f"Epoch {epoch} gen-val", gen_results)
            if use_wandb:
                wandb.log({
                    "val_gen_epoch/accuracy_strict":  gen_results["overall_strict"],
                    "val_gen_epoch/accuracy_lenient": gen_results["overall_lenient"],
                    **{f"val_gen_epoch/{ds}_strict": s["strict"]
                       for ds, s in gen_results["by_dataset"].items()},
                    **{f"val_gen_epoch/{ds}_lenient": s["lenient"]
                       for ds, s in gen_results["by_dataset"].items()},
                    "epoch": epoch,
                }, step=global_step)

        # ── Checkpointing ────────────────────────────────────────────────
        new_best = val_loss < best_val_loss
        if new_best:
            best_val_loss = val_loss
            best_dir = output_dir / "best"
            if is_main:
                best_dir.mkdir(exist_ok=True)
            accelerator.wait_for_everyone()

            if args.save_full_model:
                if is_main:
                    log.info("Saving full model to %s …", best_dir / "model")
                unwrapped = accelerator.unwrap_model(model)
                full_state = accelerator.get_state_dict(model)
                unwrapped.save_pretrained(
                    best_dir / "model",
                    is_main_process=is_main,
                    save_function=accelerator.save,
                    state_dict=full_state,
                )
                if is_main:
                    processor.save_pretrained(best_dir / "model")
            if is_main:
                log.info("New best val_loss=%.4f saved to %s", best_val_loss, best_dir)

        _save_resume_checkpoint(
            accelerator, resume_ckpt,
            next_epoch=epoch + 1, global_step=global_step,
            step_in_epoch=0,
            best_val_loss=best_val_loss,
        )

    if is_main:
        log.info("Training complete. Best val_loss: %.4f", best_val_loss)
        if use_wandb:
            wandb.finish()


def _reset_fsdp_training_state(accelerator) -> None:
    """Workaround for PyTorch FSDP bug."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._common_utils import TrainingState
        from torch.distributed.fsdp._flat_param import HandleTrainingState
    except ImportError:
        return
    for model in accelerator._models:
        for module in FSDP.fsdp_modules(model):
            module.training_state = TrainingState.IDLE
            handle = getattr(module, "_handle", None)
            if handle is not None:
                handle._training_state = HandleTrainingState.IDLE


def _save_resume_checkpoint(
    accelerator,
    ckpt_dir: Path,
    next_epoch: int,
    global_step: int,
    step_in_epoch: int,
    best_val_loss: float,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving resume checkpoint to %s …", ckpt_dir)
    _reset_fsdp_training_state(accelerator)
    accelerator.save_state(str(ckpt_dir / "accel_state"))
    if accelerator.is_main_process:
        state = {
            "next_epoch":    next_epoch,
            "global_step":   global_step,
            "step_in_epoch": step_in_epoch,
            "best_val_loss": best_val_loss,
        }
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(state, f, indent=2)
        log.info("Resume checkpoint saved (next_epoch=%d, global_step=%d)",
                 next_epoch, global_step)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline: fine-tune Qwen3.5-VL on QVID+Kinetics+HMDB51+ShareGPT4Video WITHOUT pose features"
    )
    # Model
    p.add_argument("--model_name",   default="Qwen/Qwen3.5-9B")
    p.add_argument("--output_dir",   default="/orcd/compute/ppliang/001/qwen_multi_base")
    p.add_argument("--resume_ckpt",  default="/orcd/compute/ppliang/001/qwen_multi_base/resume_ckpt")

    # Data — pose_sample_n is kept so MultiVideoDataset loads the same samples
    p.add_argument("--pose_sample_n",   type=int,   default=16)
    p.add_argument("--fps",             type=float, default=1.0)
    p.add_argument("--max_frames",      type=int,   default=8)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=4)

    # Training
    p.add_argument("--num_epochs",           type=int,   default=3)
    p.add_argument("--batch_size",           type=int,   default=1)
    p.add_argument("--grad_accum",           type=int,   default=8)
    p.add_argument("--lr",                   type=float, default=1e-5,
                   help="Learning rate (default matches train.py backbone_lr).")
    p.add_argument("--weight_decay",         type=float, default=0.05)
    p.add_argument("--warmup_ratio",         type=float, default=0.05)
    p.add_argument("--max_grad_norm",        type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--torch_compile",  action="store_true", default=False)
    p.add_argument("--no_torch_compile", dest="torch_compile", action="store_false")
    p.add_argument("--save_full_model", action="store_true", default=True)
    p.add_argument("--save_steps", type=int, default=10)
    p.add_argument("--val_steps", type=int, default=10)
    p.add_argument("--val_max_samples", type=int, default=200)
    p.add_argument("--val_max_new_tokens", type=int, default=32)

    # Logging
    p.add_argument("--log_steps",       type=int,  default=10)
    p.add_argument("--wandb_project",   type=str,  default="qwen-multi-baseline")
    p.add_argument("--wandb_run_name",  type=str,  default=None)
    p.add_argument("--no_wandb",        action="store_true")
    p.add_argument("--log_dir",         type=str, default=None,
                   help="Directory for run log files. Defaults to <output_dir>/logs.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
