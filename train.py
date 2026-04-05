#!/usr/bin/env python3
"""
train.py — Fine-tune Qwen3.5-VL on QVID + Kinetics-400 + HMDB51 with pose features.

Both the Qwen backbone AND the pose projector are trained (full fine-tuning).
Two losses are combined:
  - SFT loss    : standard causal-LM cross-entropy on answer tokens (trains backbone)
  - Pose loss   : projector predicts first answer token from last-prompt hidden +
                  pose embeddings (trains projector and aligns pose → hidden space)
  total_loss = sft_loss + args.pose_loss_weight * pose_loss

Usage
-----
  python train.py \\
      --model_name Qwen/Qwen3.5-VL-7B-Instruct \\
      --output_dir /orcd/compute/ppliang/001/qwen_multi
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from multi_dataset import get_all_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Pose projector ────────────────────────────────────────────────────────────

class PoseFeatureProjector(nn.Module):
    """2-layer MLP: pose feature dim → LLM hidden dim."""

    def __init__(self, feature_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiVideoDataset(Dataset):
    """
    Unified dataset over QVID, Kinetics-400, and HMDB51.
    Each sample is a (video_frames, pose_features, question, answer) tuple.
    """

    def __init__(
        self,
        samples: List[dict],
        fps: float = 2.0,
        max_frames: int = 32,
        pose_sample_n: int = 16,
        skip_missing_pose: bool = True,
    ):
        self.fps = fps
        self.max_frames = max_frames
        self.pose_sample_n = pose_sample_n
        self.examples: List[dict] = []

        for s in samples:
            if skip_missing_pose and not Path(s["pose_path"]).exists():
                continue
            self.examples.append(s)

        log.info("MultiVideoDataset: %d examples loaded", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        s = self.examples[idx]

        # Sample frames at target fps up to max_frames
        pil_images = _sample_frames(s["video_path"], self.fps, self.max_frames)

        # Load pre-extracted pose features
        pose_feat = _load_pose_features(s["pose_path"], self.pose_sample_n)

        return {
            "pil_images": pil_images,
            "pose_feat":  pose_feat,
            "question":   s["question"],
            "answer":     s["answer"],
            "dataset":    s["dataset"],
            "task_type":  s["task_type"],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_frames(video_path: str, fps: float, max_frames: int) -> List[Image.Image]:
    """
    Sample frames from a video at `fps` frames-per-second, capped at `max_frames`.
    Returns a variable-length list of PIL RGB images (at least 1).
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return [Image.new("RGB", (224, 224))]

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame stride in video-frame units; at least 1
    stride = max(1.0, video_fps / fps)

    # Build target indices
    indices: List[int] = []
    pos = 0.0
    while pos < total and len(indices) < max_frames:
        indices.append(min(round(pos), total - 1))
        pos += stride

    if not indices:
        indices = [0]

    indices_set = set(indices)
    collected: dict = {}
    current = 0

    while current <= max(indices_set):
        ret, frame = cap.read()
        if not ret:
            break
        if current in indices_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            collected[current] = Image.fromarray(rgb)
        current += 1

    cap.release()

    # Return in order; substitute blank for any unreadable frame
    return [
        collected.get(i, Image.new("RGB", (224, 224)))
        for i in indices
    ]


def _load_pose_features(pose_path: str, target_n: int) -> torch.Tensor:
    """Load .pt pose file and resample to target_n frames → (target_n, feat_dim)."""
    raw = torch.load(pose_path, map_location="cpu", weights_only=False)

    if isinstance(raw, torch.Tensor):
        feat = raw.float()
    elif isinstance(raw, dict):
        parts = [raw[k].float() for k in ("pose", "face", "left_hand", "right_hand") if k in raw]
        feat = torch.cat(parts, dim=-1) if parts else torch.zeros(1, 1659)
    else:
        feat = torch.zeros(target_n, 1659)

    if feat.shape[0] != target_n:
        feat = F.interpolate(
            feat.T.unsqueeze(0), size=target_n, mode="linear", align_corners=False
        ).squeeze(0).T

    return feat  # (target_n, feat_dim)


# ── Collate ───────────────────────────────────────────────────────────────────

def _collate(batch: List[dict], processor, tokenizer, embed_device: torch.device) -> dict:
    """
    Build processor inputs for a batch.

    Labels are set to -100 for all prompt tokens; answer tokens are unmasked
    so only the answer contributes to the SFT loss.
    """
    full_texts, prompt_texts, all_images, answers = [], [], [], []

    for ex in batch:
        # Build user message with video frames + question
        content = [{"type": "image"} for _ in ex["pil_images"]]
        content.append({"type": "text", "text": ex["question"]})
        user_msg = {"role": "user", "content": content}
        asst_msg = {"role": "assistant", "content": ex["answer"]}

        full_texts.append(
            processor.apply_chat_template(
                [user_msg, asst_msg], tokenize=False, add_generation_prompt=False
            )
        )
        prompt_texts.append(
            processor.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True
            )
        )
        all_images.append(ex["pil_images"])
        answers.append(ex["answer"])

    # Compute answer token lengths (text-only tokenization; no visual tokens in answer)
    answer_lens = []
    for full_t, prompt_t in zip(full_texts, prompt_texts):
        answer_text = full_t[len(prompt_t):]
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        answer_lens.append(len(answer_ids))

    # Full tokenization with visual tokens
    inputs = processor(
        text=full_texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"]  # (B, seq_len)

    # Build labels: -100 for prompt, actual IDs for answer
    labels = torch.full_like(input_ids, fill_value=-100)
    for i, ans_len in enumerate(answer_lens):
        seq_len = (input_ids[i] != tokenizer.pad_token_id).sum().item()
        # Answer occupies the last ans_len non-pad positions
        answer_start = seq_len - ans_len
        if answer_start > 0:
            labels[i, answer_start:seq_len] = input_ids[i, answer_start:seq_len]

    # Stack pose features: (B, pose_sample_n, feat_dim)
    pose_feat = torch.stack([ex["pose_feat"] for ex in batch], dim=0)

    return {
        **{k: v.to(embed_device) for k, v in inputs.items()},
        "labels":    labels.to(embed_device),
        "pose_feat": pose_feat,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        log.info("WandB run: %s", wandb.run.name)
    else:
        if not args.no_wandb and not WANDB_AVAILABLE:
            log.warning("wandb not installed — logging to console only. pip install wandb to enable.")

    # ── Resume checkpoint detection ───────────────────────────────────────────
    resume_ckpt = Path(args.resume_ckpt)
    resume_state = None
    if (resume_ckpt / "train_state.json").exists():
        with open(resume_ckpt / "train_state.json") as f:
            resume_state = json.load(f)
        log.info("Found resume checkpoint at %s (next_epoch=%d)", resume_ckpt, resume_state["next_epoch"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = str(resume_ckpt / "model") if resume_state and (resume_ckpt / "model").exists() else args.model_name
    log.info("Loading model: %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    assert hasattr(model, "model"),    "Expected model.model (backbone)"
    assert hasattr(model, "lm_head"), "Expected model.lm_head"

    # Unfreeze all backbone parameters for full fine-tuning
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    # Enable gradient checkpointing to reduce activation memory
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Total params: %s  Trainable: %s", f"{total_params:,}", f"{trainable_params:,}")

    embed_device  = next(model.model.embed_tokens.parameters()).device
    lm_head_device = next(model.lm_head.parameters()).device
    log.info("embed_device: %s  lm_head_device: %s", embed_device, lm_head_device)

    # ── Pose projector ────────────────────────────────────────────────────────
    feat_dim = (resume_state["feat_dim"] if resume_state else None) or args.pose_feature_dim
    if feat_dim is None:
        # Auto-detect from first available .pt file in splits
        splits = get_all_samples()
        for split_samples in splits.values():
            for s in split_samples:
                if Path(s["pose_path"]).exists():
                    sample_feat = _load_pose_features(s["pose_path"], 1)
                    feat_dim = sample_feat.shape[-1]
                    log.info("Auto-detected pose_feature_dim = %d", feat_dim)
                    break
            if feat_dim is not None:
                break
        if feat_dim is None:
            raise RuntimeError(
                "Could not auto-detect pose_feature_dim. "
                "Make sure pose features are extracted first, or pass --pose_feature_dim."
            )

    hidden_size = model.config.get_text_config().hidden_size
    projector = PoseFeatureProjector(feat_dim, hidden_size)
    projector = projector.to(device=lm_head_device, dtype=torch.bfloat16)
    log.info(
        "PoseFeatureProjector: %d → %d  (%s params)",
        feat_dim, hidden_size,
        f"{sum(p.numel() for p in projector.parameters()):,}",
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    if not hasattr(train, "_splits"):
        splits = get_all_samples()
    else:
        splits = train._splits  # reuse if already loaded

    train_ds = MultiVideoDataset(
        splits["train"], fps=args.fps, max_frames=args.max_frames, pose_sample_n=args.pose_sample_n
    )
    val_ds = MultiVideoDataset(
        splits["val"], fps=args.fps, max_frames=args.max_frames, pose_sample_n=args.pose_sample_n
    )

    def make_collate(d):
        return lambda b: _collate(b, processor, tokenizer, d)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=make_collate(embed_device),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=make_collate(embed_device),
    )

    # ── Optimizer — two LR groups ─────────────────────────────────────────────
    # Backbone gets backbone_lr; projector gets lr (typically 10× higher)
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params,           "lr": args.backbone_lr},
            {"params": projector.parameters(),    "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = math.ceil(len(train_loader) / args.grad_accum) * args.num_epochs
    warmup_steps = round(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1

    if resume_state:
        start_epoch   = resume_state["next_epoch"]
        global_step   = resume_state["global_step"]
        best_val_loss = resume_state["best_val_loss"]
        _load_resume_checkpoint(resume_ckpt, projector, optimizer, scheduler, lm_head_device)

    optimizer.zero_grad()

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        projector.train()

        running_loss = running_sft = running_pose = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=True)

        for step, batch in enumerate(pbar):
            pose_feat  = batch.pop("pose_feat").to(device=lm_head_device, dtype=torch.bfloat16)
            labels     = batch.pop("labels")    # on embed_device

            # ── Forward through backbone ──────────────────────────────────────
            backbone_out = model.model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
            )
            hidden = backbone_out.last_hidden_state  # (B, seq_len, H)

            # Move labels to the same device as lm_head for loss computation
            labels = labels.to(lm_head_device)
            hidden = hidden.to(lm_head_device)

            # ── SFT loss: cross-entropy on answer tokens ──────────────────────
            # We use hidden states from backbone (now WITH gradients) and
            # compute next-token prediction loss on answer positions only.
            sft_logits = model.lm_head(hidden).float()   # (B, seq_len, vocab)
            # Shift: predict token[i+1] from token[i]
            shift_logits = sft_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            sft_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # ── Pose loss: projector predicts first answer token ──────────────
            # Find the position of the first answer token per sample
            # (first position where labels != -100)
            B = hidden.shape[0]
            first_ans_pos = (labels != -100).int().argmax(dim=1).clamp(min=1)  # (B,)

            # Gather hidden state at the last prompt position (= first_ans_pos - 1)
            last_prompt_h = hidden[torch.arange(B), first_ans_pos - 1, :]  # (B, H)

            # Append pose embeddings and predict first answer token
            pose_embeds   = projector(pose_feat)                           # (B, T, H)
            extended      = torch.cat(
                [last_prompt_h.unsqueeze(1), pose_embeds], dim=1
            )                                                              # (B, 1+T, H)
            pose_logits   = model.lm_head(extended[:, -1:, :]).float()    # (B, 1, vocab)
            first_ans_ids = labels[torch.arange(B), first_ans_pos]        # (B,)

            # Only compute pose loss where a valid answer token exists
            valid_mask = first_ans_ids != -100
            if valid_mask.any():
                pose_loss = F.cross_entropy(
                    pose_logits[valid_mask, 0, :], first_ans_ids[valid_mask]
                )
            else:
                pose_loss = torch.tensor(0.0, device=lm_head_device)

            # ── Combined loss ─────────────────────────────────────────────────
            loss = sft_loss + args.pose_loss_weight * pose_loss
            (loss / args.grad_accum).backward()

            running_loss += loss.item()
            running_sft  += sft_loss.item()
            running_pose += pose_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", sft=f"{sft_loss.item():.4f}")

            if (step + 1) % args.grad_accum == 0:
                # Clip gradients across both backbone and projector
                all_params = list(model.parameters()) + list(projector.parameters())
                nn.utils.clip_grad_norm_(
                    [p for p in all_params if p.grad is not None],
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    n = args.log_steps * args.grad_accum
                    avg_loss = running_loss / n
                    avg_sft  = running_sft  / n
                    avg_pose = running_pose / n
                    running_loss = running_sft = running_pose = 0.0

                    backbone_lr = scheduler.get_last_lr()[0]
                    proj_lr     = scheduler.get_last_lr()[1]

                    log.info(
                        "epoch %d  step %d  loss %.4f  sft %.4f  pose %.4f"
                        "  backbone_lr %.2e  proj_lr %.2e",
                        epoch, global_step, avg_loss, avg_sft, avg_pose,
                        backbone_lr, proj_lr,
                    )
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        sft=f"{avg_sft:.4f}",
                        pose=f"{avg_pose:.4f}",
                    )

                    if use_wandb:
                        wandb.log({
                            "train/loss":       avg_loss,
                            "train/sft_loss":   avg_sft,
                            "train/pose_loss":  avg_pose,
                            "train/backbone_lr": backbone_lr,
                            "train/proj_lr":    proj_lr,
                            "epoch":            epoch,
                        }, step=global_step)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        projector.eval()
        val_loss = val_sft = val_pose = 0.0
        correct = total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                pose_feat = batch.pop("pose_feat").to(device=lm_head_device, dtype=torch.bfloat16)
                labels    = batch.pop("labels").to(lm_head_device)

                backbone_out = model.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    pixel_values_videos=batch.get("pixel_values_videos"),
                    video_grid_thw=batch.get("video_grid_thw"),
                )
                hidden = backbone_out.last_hidden_state.to(lm_head_device)

                sft_logits = model.lm_head(hidden).float()
                sft_loss_v = F.cross_entropy(
                    sft_logits[:, :-1, :].contiguous().view(-1, sft_logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )

                B = hidden.shape[0]
                first_ans_pos = (labels != -100).int().argmax(dim=1).clamp(min=1)
                last_prompt_h = hidden[torch.arange(B), first_ans_pos - 1, :]
                pose_embeds   = projector(pose_feat)
                extended      = torch.cat([last_prompt_h.unsqueeze(1), pose_embeds], dim=1)
                pose_logits_v = model.lm_head(extended[:, -1:, :]).float()
                first_ans_ids = labels[torch.arange(B), first_ans_pos]
                valid_mask    = first_ans_ids != -100

                if valid_mask.any():
                    pose_loss_v = F.cross_entropy(
                        pose_logits_v[valid_mask, 0, :], first_ans_ids[valid_mask]
                    )
                    preds   = pose_logits_v[valid_mask, 0, :].argmax(dim=-1)
                    correct += (preds == first_ans_ids[valid_mask]).sum().item()
                    total   += valid_mask.sum().item()
                else:
                    pose_loss_v = torch.tensor(0.0)

                loss_v  = sft_loss_v + args.pose_loss_weight * pose_loss_v
                val_loss += loss_v.item()
                val_sft  += sft_loss_v.item()
                val_pose += pose_loss_v.item()

        n_val = max(1, len(val_loader))
        val_loss /= n_val
        val_sft  /= n_val
        val_pose /= n_val
        val_acc   = correct / max(1, total)

        log.info(
            "=== Epoch %d  val_loss=%.4f  val_sft=%.4f  val_pose=%.4f  val_acc=%.4f ===",
            epoch, val_loss, val_sft, val_pose, val_acc,
        )

        if use_wandb:
            wandb.log({
                "val/loss":      val_loss,
                "val/sft_loss":  val_sft,
                "val/pose_loss": val_pose,
                "val/accuracy":  val_acc,
                "epoch":         epoch,
            }, step=global_step)

        # ── Checkpointing ─────────────────────────────────────────────────────
        # Save projector weights every epoch (small, fast)
        ckpt_dir = output_dir / f"epoch-{epoch}"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(projector.state_dict(), ckpt_dir / "pose_projector.pt")
        _save_config(ckpt_dir, feat_dim, hidden_size, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True)
            torch.save(projector.state_dict(), best_dir / "pose_projector.pt")
            _save_config(best_dir, feat_dim, hidden_size, args)
            # Save full model at best checkpoint (large — ~14 GB for 7B)
            if args.save_full_model:
                log.info("Saving full model to %s …", best_dir / "model")
                model.save_pretrained(best_dir / "model")
                processor.save_pretrained(best_dir / "model")
            log.info("New best val_loss=%.4f saved to %s", best_val_loss, best_dir)

        # Save full resume checkpoint (model + projector + optimizer + scheduler)
        _save_resume_checkpoint(
            resume_ckpt, model, projector, optimizer, scheduler, processor,
            next_epoch=epoch + 1, global_step=global_step,
            best_val_loss=best_val_loss, feat_dim=feat_dim, hidden_size=hidden_size,
        )

    log.info("Training complete. Best val_loss: %.4f", best_val_loss)
    if use_wandb:
        wandb.finish()


def _save_config(directory: Path, feat_dim: int, hidden_size: int, args) -> None:
    cfg = {
        "pose_feature_dim": feat_dim,
        "hidden_size":      hidden_size,
        "pose_sample_n":    args.pose_sample_n,
        "model_name":       args.model_name,
        "backbone_lr":      args.backbone_lr,
        "lr":               args.lr,
        "pose_loss_weight": args.pose_loss_weight,
    }
    with open(directory / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _save_resume_checkpoint(
    ckpt_dir: Path,
    model,
    projector: PoseFeatureProjector,
    optimizer,
    scheduler,
    processor,
    next_epoch: int,
    global_step: int,
    best_val_loss: float,
    feat_dim: int,
    hidden_size: int,
) -> None:
    """Save everything needed to resume training from the next epoch."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving resume checkpoint to %s …", ckpt_dir)
    model.save_pretrained(ckpt_dir / "model")
    processor.save_pretrained(ckpt_dir / "model")
    torch.save(projector.state_dict(), ckpt_dir / "pose_projector.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    state = {
        "next_epoch":    next_epoch,
        "global_step":   global_step,
        "best_val_loss": best_val_loss,
        "feat_dim":      feat_dim,
        "hidden_size":   hidden_size,
    }
    with open(ckpt_dir / "train_state.json", "w") as f:
        json.dump(state, f, indent=2)
    log.info("Resume checkpoint saved (next_epoch=%d, global_step=%d)", next_epoch, global_step)


def _load_resume_checkpoint(
    ckpt_dir: Path,
    projector: PoseFeatureProjector,
    optimizer,
    scheduler,
    lm_head_device: torch.device,
) -> dict:
    """Load projector, optimizer, and scheduler states; return train_state dict."""
    log.info("Loading resume checkpoint from %s …", ckpt_dir)
    with open(ckpt_dir / "train_state.json") as f:
        state = json.load(f)
    projector.load_state_dict(
        torch.load(ckpt_dir / "pose_projector.pt", map_location=lm_head_device, weights_only=True)
    )
    opt_state = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(opt_state)
    scheduler.load_state_dict(
        torch.load(ckpt_dir / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    log.info(
        "Resumed: next_epoch=%d  global_step=%d  best_val_loss=%.4f",
        state["next_epoch"], state["global_step"], state["best_val_loss"],
    )
    return state


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Qwen3.5-VL on QVID+Kinetics+HMDB51 with pose features"
    )
    # Model
    p.add_argument("--model_name",   default="Qwen/Qwen3.5-9B")
    p.add_argument("--output_dir",   default="/orcd/compute/ppliang/001/qwen_multi")
    p.add_argument("--resume_ckpt",  default="/home/ixzhu/orcd/scratch/qwen_pose/resume_ckpt",
                   help="Directory to save/load full resume checkpoints (model + optimizer + scheduler).")

    # Data
    p.add_argument("--pose_feature_dim", type=int, default=None,
                   help="Auto-detected from first .pt file if not set.")
    p.add_argument("--pose_sample_n",   type=int,   default=16)
    p.add_argument("--fps",             type=float, default=2.0,
                   help="Target sampling rate (frames/sec) for video frames fed to the vision encoder.")
    p.add_argument("--max_frames",      type=int,   default=32,
                   help="Maximum number of video frames per clip (caps fps-based sampling).")
    p.add_argument("--seed",            type=int,   default=42)

    # Training
    p.add_argument("--num_epochs",           type=int,   default=3)
    p.add_argument("--batch_size",           type=int,   default=1)
    p.add_argument("--grad_accum",           type=int,   default=8)
    p.add_argument("--backbone_lr",          type=float, default=1e-5,
                   help="LR for Qwen backbone parameters.")
    p.add_argument("--lr",                   type=float, default=1e-4,
                   help="LR for pose projector parameters.")
    p.add_argument("--weight_decay",         type=float, default=0.05)
    p.add_argument("--warmup_ratio",         type=float, default=0.05)
    p.add_argument("--max_grad_norm",        type=float, default=1.0)
    p.add_argument("--pose_loss_weight",     type=float, default=0.1,
                   help="Weight for pose auxiliary loss (default: 0.1).")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--save_full_model", action="store_true", default=False,
                   help="Save full Qwen model at best checkpoint (~14 GB).")

    # Logging
    p.add_argument("--log_steps",       type=int,  default=10)
    p.add_argument("--wandb_project",   type=str,  default="qwen-multi-video")
    p.add_argument("--wandb_run_name",  type=str,  default=None)
    p.add_argument("--no_wandb",        action="store_true",
                   help="Disable WandB logging.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
