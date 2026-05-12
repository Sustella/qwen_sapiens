#!/usr/bin/env python3
"""
train_avasd.py — Fine-tune the pose-augmented Qwen3.5-VL checkpoint on AV-ASD.

Starting point: the best pose-augmented checkpoint saved by train.py
(i.e. ``/orcd/compute/ppliang/001/qwen_multi/best`` by default).  The pose
projector is frozen; only the backbone + lm_head are trained.  Pose features
are injected at the prompt/answer boundary via the same forward-hook mechanism
used in train.py, so the model keeps the pose-conditioning it learnt during
pre-finetuning.

Logging mirrors train.py (wandb + stdout + timestamped log file), including
per-step train loss, per-epoch validation loss, per-behavior generative
accuracy, and balanced accuracy / F1 (matching evaluate_avasd.py's metrics).

Usage
-----
  # Single-GPU
  python train_avasd.py \\
      --init_checkpoint /orcd/compute/ppliang/001/qwen_multi/best \\
      --output_dir /orcd/compute/ppliang/001/qwen_avasd

  # FSDP multi-GPU
  accelerate launch train_avasd.py \\
      --init_checkpoint /orcd/compute/ppliang/001/qwen_multi/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B \\
      --output_dir /orcd/compute/ppliang/001/qwen_avasd
"""

import argparse
import functools
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# Same FSDP StopIteration workaround as train.py
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

from train import (
    PoseFeatureProjector,
    MultiVideoDataset,
    _collate,
    _collate_gen,
    _make_pose_hook,
    _get_language_model,
    _save_resume_checkpoint,
    _reset_fsdp_training_state,
)
from avasd_dataset import get_avasd_samples, BEHAVIORS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _detect_checkpoint_format(checkpoint_dir: str) -> str:
    ckpt = Path(checkpoint_dir)
    if (ckpt / "model").is_dir():
        return "hf"
    if (ckpt / "accel_state").is_dir():
        return "accel"
    raise RuntimeError(
        f"Cannot detect checkpoint format at {checkpoint_dir} — expected "
        "either a model/ (HF) or an accel_state/ (accelerate) subdirectory."
    )


def _load_model_from_checkpoint(init_ckpt: str, base_model: str):
    """Load the pose-augmented backbone from an HF or accel checkpoint.

    Returns (model, processor).  The model lives on CPU; FSDP prepare() will
    place it on the right device later.
    """
    fmt = _detect_checkpoint_format(init_ckpt)
    ckpt = Path(init_ckpt)

    if fmt == "hf":
        model_dir = ckpt / "model"
        log.info("Loading HF-format backbone from %s", model_dir)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        return model, processor

    # Accelerate format — use evaluate_multidataset helpers
    log.info("Loading accel-format backbone (base=%s) from %s", base_model, ckpt)
    from evaluate_multidataset import (
        load_model_from_accel_checkpoint,
    )
    model, processor, _ = load_model_from_accel_checkpoint(str(ckpt), base_model)
    # Move back to CPU so the accelerator can place / FSDP-wrap it cleanly.
    model = model.to("cpu")
    for p in model.parameters():
        p.requires_grad = True
    return model, processor


def _load_projector_frozen(init_ckpt: str, device) -> PoseFeatureProjector:
    """Load the pose projector, freeze it, move to device/bf16."""
    from evaluate_multidataset import (
        load_projector_from_hf,
        load_projector_from_accel,
    )
    fmt = _detect_checkpoint_format(init_ckpt)
    if fmt == "hf":
        projector = load_projector_from_hf(init_ckpt, device)
    else:
        projector = load_projector_from_accel(init_ckpt, device)

    if projector is None:
        raise RuntimeError(
            f"No pose projector found at {init_ckpt}.  train_avasd.py needs "
            "the projector weights to inject pose tokens — check that "
            "pose_projector.pt or pytorch_model_fsdp_1/ exists under the "
            "checkpoint directory."
        )

    for p in projector.parameters():
        p.requires_grad = False
    projector.eval()
    return projector


# ── Answer scoring (binary) ───────────────────────────────────────────────────

def _parse_binary(pred: str) -> Optional[int]:
    pred = pred.strip()
    if not pred:
        return None
    for ch in pred:
        if ch == "0":
            return 0
        if ch == "1":
            return 1
    try:
        v = int(pred)
        if v in (0, 1):
            return v
    except ValueError:
        pass
    return None


def _compute_binary_metrics(y_true: List[int], y_pred: List[Optional[int]]) -> dict:
    tp = fp = fn = tn = 0
    unparsable = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            unparsable += 1
            # Count unparsable as wrong — follow evaluate_avasd.py convention
            if t == 1:
                fn += 1
            else:
                fp += 1
            continue
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total, "unparsable": unparsable,
        "accuracy": acc,
        "balanced_accuracy": (tpr + tnr) / 2,
        "precision": prec, "recall": tpr, "f1": f1,
        "tpr": tpr, "tnr": tnr,
    }


# ── AV-ASD dataset wrapper ────────────────────────────────────────────────────

class AVASDDataset(MultiVideoDataset):
    """MultiVideoDataset subclass that also surfaces ``behavior`` so the
    gen-val loop can report per-behavior accuracy."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["behavior"] = self.examples[idx]["behavior"]
        return item


def _collate_train(batch, processor, tokenizer, pose_sample_n):
    """Same as train._collate but preserves ``behavior`` for logging."""
    return _collate(batch, processor, tokenizer, pose_sample_n)


def _collate_gen_avasd(batch, processor, tokenizer, pose_sample_n):
    """Same as train._collate_gen but also carries behavior labels."""
    result = _collate_gen(batch, processor, tokenizer, pose_sample_n)
    result["behaviors"] = [ex["behavior"] for ex in batch]
    return result


# ── Generative validation (AV-ASD) ───────────────────────────────────────────

@torch.no_grad()
def _run_generative_validation_avasd(
    model, projector, val_ds, processor, tokenizer, accelerator,
    pose_sample_n: int, max_new_tokens: int = 4, max_samples: int = 0,
) -> dict:
    """Run generation over the val set, compute per-behavior binary metrics."""
    is_main = accelerator.is_main_process

    if max_samples and max_samples < len(val_ds):
        # Stratify by behavior so every one is represented
        by_b = defaultdict(list)
        for idx in range(len(val_ds)):
            by_b[val_ds.examples[idx]["behavior"]].append(idx)
        total = sum(len(v) for v in by_b.values())
        selected = []
        for b, ids in sorted(by_b.items()):
            n = max(1, round(max_samples * len(ids) / total))
            step = max(1, len(ids) // n)
            selected.extend(ids[::step][:n])
        subset = torch.utils.data.Subset(val_ds, selected[:max_samples])
    else:
        subset = val_ds

    collate_fn = lambda b: _collate_gen_avasd(b, processor, tokenizer, pose_sample_n)
    gen_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
    gen_loader = accelerator.prepare(gen_loader)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    lang_model = _get_language_model(model, accelerator)

    by_behavior = defaultdict(lambda: {"y_true": [], "y_pred": []})
    all_true: List[int] = []
    all_pred: List[Optional[int]] = []

    was_training = model.training
    model.eval()

    for batch in tqdm(gen_loader, desc="Gen-val", leave=False, disable=not is_main):
        answers   = batch.pop("answers")
        behaviors = batch.pop("behaviors")
        pose_feat = batch.pop("pose_feat").to(dtype=torch.bfloat16)
        pose_positions = batch.pop("pose_positions")
        # task_types / datasets not needed here
        batch.pop("datasets", None)
        batch.pop("task_types", None)

        pose_embeds = projector(pose_feat.to(next(projector.parameters()).device))
        hook_fn = _make_pose_hook(pose_embeds, pose_positions)
        handle = lang_model.register_forward_pre_hook(hook_fn, with_kwargs=True)

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
            handle.remove()
            continue
        handle.remove()

        preds_text = [tokenizer.decode(ids, skip_special_tokens=True).strip()
                      for ids in new_ids]

        for pred_t, gt, b in zip(preds_text, answers, behaviors):
            pred = _parse_binary(pred_t)
            true = int(gt)
            by_behavior[b]["y_true"].append(true)
            by_behavior[b]["y_pred"].append(pred)
            all_true.append(true)
            all_pred.append(pred)

    if was_training:
        model.train()

    per_behavior = {
        b: _compute_binary_metrics(d["y_true"], d["y_pred"])
        for b, d in sorted(by_behavior.items())
    }
    overall = _compute_binary_metrics(all_true, all_pred)

    return {"overall": overall, "by_behavior": per_behavior}


def _log_avasd_results(label: str, results: dict, log_fn=log.info):
    ov = results["overall"]
    log_fn(
        "%s: acc=%.4f  bal_acc=%.4f  f1=%.4f  (tp=%d fp=%d fn=%d tn=%d unpars=%d)",
        label, ov["accuracy"], ov["balanced_accuracy"], ov["f1"],
        ov["tp"], ov["fp"], ov["fn"], ov["tn"], ov["unparsable"],
    )
    for b, m in results["by_behavior"].items():
        log_fn(
            "    %-45s acc=%.4f  bal=%.4f  f1=%.4f  (%d/%d)",
            b, m["accuracy"], m["balanced_accuracy"], m["f1"],
            m["tp"] + m["tn"], m["total"],
        )


def _results_to_wandb(prefix: str, results: dict) -> dict:
    metrics = {
        f"{prefix}/accuracy":         results["overall"]["accuracy"],
        f"{prefix}/balanced_accuracy":results["overall"]["balanced_accuracy"],
        f"{prefix}/f1":               results["overall"]["f1"],
        f"{prefix}/precision":        results["overall"]["precision"],
        f"{prefix}/recall":           results["overall"]["recall"],
        f"{prefix}/unparsable":       results["overall"]["unparsable"],
    }
    for b, m in results["by_behavior"].items():
        safe = b.replace(" ", "_").replace("/", "_")
        metrics[f"{prefix}_behavior/{safe}_acc"] = m["accuracy"]
        metrics[f"{prefix}_behavior/{safe}_bal_acc"] = m["balanced_accuracy"]
        metrics[f"{prefix}_behavior/{safe}_f1"] = m["f1"]
    return metrics


# ── Save helpers ─────────────────────────────────────────────────────────────

def _save_avasd_config(directory: Path, args, init_ckpt_cfg: dict) -> None:
    cfg = {
        "init_checkpoint":   args.init_checkpoint,
        "base_model":        args.base_model,
        "pose_feature_dim":  init_ckpt_cfg.get("pose_feature_dim"),
        "hidden_size":       init_ckpt_cfg.get("hidden_size"),
        "pose_sample_n":     args.pose_sample_n,
        "backbone_lr":       args.backbone_lr,
        "anonymized":        args.anonymized,
    }
    with open(directory / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


# ── Training ──────────────────────────────────────────────────────────────────

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
            size_based_auto_wrap_policy, min_num_params=int(1e8),
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

    # ── Log file (main process only) ─────────────────────────────────────────
    if is_main:
        log_dir = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"avasd_run_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"),
        )
        logging.getLogger().addHandler(file_handler)
        log.info("Logging to %s", log_file)

    log.info("Accelerator: num_processes=%d  device=%s",
             accelerator.num_processes, device)

    # ── WandB ─────────────────────────────────────────────────────────────────
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

    # ── Load backbone + projector ────────────────────────────────────────────
    log.info("Loading init checkpoint: %s", args.init_checkpoint)
    model, processor = _load_model_from_checkpoint(args.init_checkpoint, args.base_model)
    tokenizer = processor.tokenizer

    for p in model.parameters():
        p.requires_grad = True
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        log.info("Gradient checkpointing enabled.")

    # Load the projector on CPU first; we'll move to the right device after
    # accelerator.prepare runs.
    projector = _load_projector_frozen(args.init_checkpoint, device="cpu")
    # Load config so we can stash it in the output checkpoints.
    init_cfg_path = Path(args.init_checkpoint) / "config.json"
    if init_cfg_path.exists():
        with open(init_cfg_path) as f:
            init_ckpt_cfg = json.load(f)
    else:
        # Pull from train_state.json (accel format)
        ts_path = Path(args.init_checkpoint) / "train_state.json"
        init_ckpt_cfg = {}
        if ts_path.exists():
            with open(ts_path) as f:
                ts = json.load(f)
            init_ckpt_cfg = {
                "pose_feature_dim": ts.get("feat_dim"),
                "hidden_size":      ts.get("hidden_size"),
            }

    log.info(
        "Backbone params: %s  (trainable)",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    log.info(
        "Projector params: %s  (frozen)",
        f"{sum(p.numel() for p in projector.parameters()):,}",
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    splits = get_avasd_samples(
        splits=["train", "val"],
        anonymized=args.anonymized,
        require_video=True,
    )
    train_ds = AVASDDataset(
        splits["train"], fps=args.fps, max_frames=args.max_frames,
        pose_sample_n=args.pose_sample_n,
    )
    val_ds = AVASDDataset(
        splits["val"], fps=args.fps, max_frames=args.max_frames,
        pose_sample_n=args.pose_sample_n,
    )

    collate_fn = lambda b: _collate_train(b, processor, tokenizer, args.pose_sample_n)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    # ── Prepare (FSDP wrap) ──────────────────────────────────────────────────
    model, train_loader, val_loader = accelerator.prepare(
        model, train_loader, val_loader,
    )
    # Projector stays unwrapped so we can freely call projector(pose_feat).
    projector = projector.to(device=device, dtype=torch.bfloat16)

    if args.torch_compile:
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)

    model_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.backbone_lr,
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

    model_scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda)
    model_optimizer, model_scheduler = accelerator.prepare(
        model_optimizer, model_scheduler,
    )

    # ── Output dirs ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0

    model_optimizer.zero_grad()
    lang_model = _get_language_model(model, accelerator)

    # Warm up FSDP with a dummy forward (see train.py for rationale)
    if is_main:
        log.info("Initializing FSDP …")
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        model(input_ids=_dummy, attention_mask=torch.ones_like(_dummy))
    accelerator.wait_for_everyone()

    # ── Pre-training baseline on val ─────────────────────────────────────────
    if is_main:
        log.info("Evaluating starting checkpoint on AV-ASD val …")
    baseline = _run_generative_validation_avasd(
        model, projector, val_ds, processor, tokenizer, accelerator,
        pose_sample_n=args.pose_sample_n,
        max_new_tokens=args.val_max_new_tokens,
        max_samples=args.val_max_samples,
    )
    if is_main:
        _log_avasd_results("Baseline (starting ckpt, pose-projector frozen)", baseline)
        if use_wandb:
            wandb.log(_results_to_wandb("baseline_val", baseline), step=0)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
            leave=True, disable=not is_main,
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                pose_feat      = batch.pop("pose_feat").to(dtype=torch.bfloat16)
                labels         = batch.pop("labels")
                pose_positions = batch.pop("pose_positions")

                with torch.no_grad():
                    pose_embeds = projector(pose_feat.to(
                        next(projector.parameters()).device
                    ))

                hook_fn = _make_pose_hook(pose_embeds, pose_positions)
                handle = lang_model.register_forward_pre_hook(hook_fn, with_kwargs=True)

                backbone_out = model.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    pixel_values_videos=batch.get("pixel_values_videos"),
                    video_grid_thw=batch.get("video_grid_thw"),
                    mm_token_type_ids=batch.get("mm_token_type_ids"),
                )
                handle.remove()

                hidden = backbone_out.last_hidden_state.to(torch.bfloat16)
                logits = model.lm_head(hidden).float()
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(model.parameters()), args.max_grad_norm,
                    )

                model_optimizer.step()
                model_scheduler.step()
                model_optimizer.zero_grad()

            running_loss += loss.item()
            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if accelerator.sync_gradients:
                global_step += 1
                step_metrics = {"epoch": epoch} if use_wandb else {}

                if global_step % args.save_steps == 0:
                    torch.cuda.synchronize()
                    _save_resume_checkpoint(
                        accelerator, Path(args.resume_ckpt),
                        next_epoch=epoch, global_step=global_step,
                        step_in_epoch=step + 1,
                        best_val_loss=best_val_loss,
                        feat_dim=init_ckpt_cfg.get("pose_feature_dim", 0),
                        hidden_size=init_ckpt_cfg.get("hidden_size", 0),
                    )

                    if args.val_steps > 0 and global_step % args.val_steps == 0:
                        if is_main:
                            log.info("Running gen-val at step %d …", global_step)
                        gen_results = _run_generative_validation_avasd(
                            model, projector, val_ds, processor, tokenizer, accelerator,
                            pose_sample_n=args.pose_sample_n,
                            max_new_tokens=args.val_max_new_tokens,
                            max_samples=args.val_max_samples,
                        )
                        if is_main:
                            _log_avasd_results(f"  gen-val step {global_step}", gen_results)
                            step_metrics.update(_results_to_wandb("val_gen", gen_results))

                if global_step % args.log_steps == 0 and is_main:
                    n = args.log_steps * args.grad_accum
                    avg_loss = running_loss / n
                    running_loss = 0.0
                    backbone_lr = model_scheduler.get_last_lr()[0]
                    log.info(
                        "epoch %d  step %d  loss %.4f  backbone_lr %.2e",
                        epoch, global_step, avg_loss, backbone_lr,
                    )
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    step_metrics.update({
                        "train/loss":        avg_loss,
                        "train/backbone_lr": backbone_lr,
                    })

                if use_wandb and step_metrics and is_main:
                    wandb.log(step_metrics, step=global_step)

        # ── Validation loss ──────────────────────────────────────────────────
        model.eval()
        val_loss_sum = torch.zeros(1, device=device)
        n_batches    = torch.zeros(1, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False,
                              disable=not is_main):
                pose_feat      = batch.pop("pose_feat").to(dtype=torch.bfloat16)
                labels         = batch.pop("labels")
                pose_positions = batch.pop("pose_positions")

                pose_embeds = projector(pose_feat.to(
                    next(projector.parameters()).device
                ))
                hook_fn = _make_pose_hook(pose_embeds, pose_positions)
                handle = lang_model.register_forward_pre_hook(hook_fn, with_kwargs=True)

                backbone_out = model.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    pixel_values_videos=batch.get("pixel_values_videos"),
                    video_grid_thw=batch.get("video_grid_thw"),
                    mm_token_type_ids=batch.get("mm_token_type_ids"),
                )
                handle.remove()

                hidden = backbone_out.last_hidden_state.to(torch.bfloat16)
                logits = model.lm_head(hidden).float()
                loss_v = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                val_loss_sum += loss_v.detach()
                n_batches    += 1

        val_loss_sum = accelerator.reduce(val_loss_sum, reduction="sum")
        n_batches    = accelerator.reduce(n_batches,    reduction="sum")
        val_loss = (val_loss_sum / n_batches.clamp(min=1)).item()

        if is_main:
            log.info("=== Epoch %d  val_loss=%.4f ===", epoch, val_loss)
            if use_wandb:
                wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        # ── End-of-epoch gen-val (full set) ──────────────────────────────────
        if is_main:
            log.info("Running end-of-epoch gen-val …")
        gen_results = _run_generative_validation_avasd(
            model, projector, val_ds, processor, tokenizer, accelerator,
            pose_sample_n=args.pose_sample_n,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=0,
        )
        if is_main:
            _log_avasd_results(f"Epoch {epoch} gen-val", gen_results)
            if use_wandb:
                wandb.log(
                    {**_results_to_wandb("val_gen_epoch", gen_results), "epoch": epoch},
                    step=global_step,
                )

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt_dir = output_dir / f"epoch-{epoch}"
        if is_main:
            ckpt_dir.mkdir(exist_ok=True)
        accelerator.wait_for_everyone()

        new_best = val_loss < best_val_loss
        best_dir = output_dir / "best"
        if new_best:
            best_val_loss = val_loss
            if is_main:
                best_dir.mkdir(exist_ok=True)
                _save_avasd_config(best_dir, args, init_ckpt_cfg)
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
                    # Copy the (frozen) projector + its config so downstream
                    # scripts can load the full pipeline from this dir.
                    torch.save(
                        {k: v.detach().cpu() for k, v in projector.state_dict().items()},
                        best_dir / "pose_projector.pt",
                    )
            if is_main:
                log.info("New best val_loss=%.4f saved to %s", best_val_loss, best_dir)

        _save_resume_checkpoint(
            accelerator, Path(args.resume_ckpt),
            next_epoch=epoch + 1, global_step=global_step,
            step_in_epoch=0,
            best_val_loss=best_val_loss,
            feat_dim=init_ckpt_cfg.get("pose_feature_dim", 0),
            hidden_size=init_ckpt_cfg.get("hidden_size", 0),
        )

    if is_main:
        log.info("Training complete. Best val_loss: %.4f", best_val_loss)
        if use_wandb:
            wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune pose-augmented Qwen on AV-ASD (projector frozen)."
    )
    # Init checkpoint
    p.add_argument(
        "--init_checkpoint",
        default="/orcd/compute/ppliang/001/qwen_multi/best",
        help="HF or accelerate checkpoint directory saved by train.py.",
    )
    p.add_argument(
        "--base_model", default="Qwen/Qwen3.5-9B",
        help="Required when --init_checkpoint is an accelerate checkpoint.",
    )

    # Output
    p.add_argument("--output_dir",  default="/orcd/compute/ppliang/001/qwen_avasd")
    p.add_argument("--resume_ckpt", default="/orcd/compute/ppliang/001/qwen_avasd/resume_ckpt")

    # Data
    p.add_argument("--pose_sample_n", type=int, default=16)
    p.add_argument("--fps",           type=float, default=1.0)
    p.add_argument("--max_frames",    type=int, default=8)
    p.add_argument("--anonymized",    action="store_true", default=False,
                   help="Use anonymized clips + mesh prompt (matches evaluate_avasd.py).")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--num_workers",   type=int, default=4)

    # Training
    p.add_argument("--num_epochs",    type=int, default=3)
    p.add_argument("--batch_size",    type=int, default=1)
    p.add_argument("--grad_accum",    type=int, default=8)
    p.add_argument("--backbone_lr",   type=float, default=1e-5)
    p.add_argument("--weight_decay",  type=float, default=0.05)
    p.add_argument("--warmup_ratio",  type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--torch_compile", action="store_true", default=False)
    p.add_argument("--save_full_model", action="store_true", default=True)
    p.add_argument("--save_steps",    type=int, default=20)
    p.add_argument("--val_steps",     type=int, default=40)
    p.add_argument("--val_max_samples", type=int, default=180,
                   help="Stratified-by-behavior cap for mid-training gen-val "
                        "(0 = full val set, same as end-of-epoch).")
    p.add_argument("--val_max_new_tokens", type=int, default=4)

    # Logging
    p.add_argument("--log_steps",      type=int, default=5)
    p.add_argument("--wandb_project",  type=str, default="qwen-avasd-finetune")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb",       action="store_true")
    p.add_argument(
        "--log_dir", type=str,
        default="/orcd/compute/ppliang/001/qwen_avasd/logs",
    )

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
