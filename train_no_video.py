#!/usr/bin/env python3
"""
train_no_video.py — Continue training the train.py pose checkpoint with the
video tokens removed from the model input.

Mirrors ../train.py except for these three deltas:
  1. Load fine-tuned backbone + projector from `--pose_ckpt/` (model dir +
     pose_projector.pt + config.json) instead of the base model + a fresh
     projector.
  2. Skip {"type": "image"} content items and `images=` in the collate so the
     processor never emits visual tokens — the model sees only text + pose.
  3. Add `--freeze_backbone` for an ablation that trains only the projector.

Pose features can come from either source, transparently per-sample (same as
train.py):
  • offline cache file at ``_per_frame_cache_path(s, fps, max_frames)``
    (produced by ``extract_poses_per_frame.py``) — loaded directly.
  • otherwise extracted ONLINE in the DataLoader worker via MediaPipe on
    sampled video frames.
The video frames themselves are never sent to the model.

Usage
-----
  python train_no_video/train_no_video.py \\
      --pose_ckpt   /orcd/compute/ppliang/001/qwen_multi/best \\
      --output_dir  /orcd/compute/ppliang/001/qwen_pose_only
"""

import argparse
import functools
import json
import logging
import math
import os

# Silence MediaPipe / TFLite stderr spam so the tqdm bar isn't drowned out
# by warnings during pose extraction. Must be set before mediapipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")        # absl/glog: ERROR+ only
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # TFLite XNNPACK chatter

import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

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
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Make sibling modules importable when running this file directly from its dir.
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_PARENT / "utils") not in sys.path:
    sys.path.insert(0, str(_PARENT / "utils"))

from multi_dataset import get_all_samples
from pose_features import LandmarkerManager, TOTAL_DIM as POSE_TOTAL_DIM  # noqa: E402


# ── Online pose extractor — same as train.py ─────────────────────────────────
# Mirrors the offline `extract_features_for_frames` from utils/pose_features.py:
# fresh `LandmarkerManager(running_mode='video')` per video, monotonically
# increasing 33ms timestamps, same feature concatenation order. The only
# difference vs offline is *which* frames are fed in — here it's the same
# PIL frames sampled in __getitem__, processed in temporal order.
POSE_FEATURE_TYPES = ["pose", "face", "left_hand", "right_hand"]


def _extract_pose_for_frames(pil_images: List[Image.Image]) -> torch.Tensor:
    """Run MediaPipe on each PIL frame in temporal order with a fresh per-video
    LandmarkerManager (running_mode='video'); return (N_frames, 1659) float
    tensor. Concatenation order: pose(33*3) + face(478*3) + L_hand(21*3) +
    R_hand(21*3) = 1659.
    """
    import numpy as np
    lm = LandmarkerManager(
        feature_types=POSE_FEATURE_TYPES,
        running_mode="video",
    )
    try:
        rows = []
        for img in pil_images:
            rgb = np.asarray(img.convert("RGB"))
            feats = lm.process(rgb)
            rows.append(np.concatenate([
                feats.get("pose",       np.zeros(33 * 3,  dtype=np.float32)),
                feats.get("face",       np.zeros(478 * 3, dtype=np.float32)),
                feats.get("left_hand",  np.zeros(21 * 3,  dtype=np.float32)),
                feats.get("right_hand", np.zeros(21 * 3,  dtype=np.float32)),
            ]))
    finally:
        lm.close()
    return torch.from_numpy(np.stack(rows, axis=0)).float()  # (N, 1659)


def _per_frame_cache_path(s: dict, fps: float, max_frames: int) -> Path:
    """Cache path for offline-extracted per-frame pose features. Keyed by fps
    and max_frames so different sampling configs don't collide."""
    p = Path(s["pose_path"])
    return p.with_name(p.stem + f".perframe_fps{fps:.2f}_max{max_frames}.pt")


def _is_video_readable(video_path: str) -> bool:
    """Quick check: cv2 can open the file and it has at least one frame."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        cap.release()


# ── Answer matching (mirrors evaluate_multidataset.py) ───────────────────────

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


# ── Pose projector (same as train.py) ────────────────────────────────────────

class PoseFeatureProjector(nn.Module):
    """Per-frame pose projector matching train.py's signature exactly so the
    saved pose_projector.pt loads cleanly.

    Forward input  : (B, N_frames, feature_dim)
    Forward output : (B, N_frames, tokens_per_frame, hidden_size)
    """

    def __init__(self, feature_dim: int, hidden_size: int, tokens_per_frame: int = 16):
        super().__init__()
        self.feature_dim      = feature_dim
        self.hidden_size      = hidden_size
        self.tokens_per_frame = tokens_per_frame
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, tokens_per_frame * hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        return out.view(*x.shape[:-1], self.tokens_per_frame, self.hidden_size)


# ── Dataset (decodes frames for online pose extraction) ──────────────────────

class MultiVideoDataset(Dataset):
    """Same dataset as train.py: samples video frames at the given fps and
    obtains per-frame pose features. Pose features may come from either
    source, transparently per-sample:
      • offline cache file at ``_per_frame_cache_path(s, fps, max_frames)``
        (produced by ``extract_poses_per_frame.py``) — loaded directly.
      • otherwise extracted online in the DataLoader worker via MediaPipe
        on the sampled PIL frames.

    Either way the resulting tensor has shape ``(N_frames, 1659)`` aligned
    1-to-1 with the frames in ``pil_images``. The frames are returned in the
    item dict but the no-video collate ignores them, so they never reach the
    model.

    Unreadable videos are dropped at init via a one-time pre-scan with a
    progress bar (mirrors train.py's behavior).
    """

    def __init__(
        self,
        samples: List[dict],
        fps: float = 2.0,
        max_frames: int = 32,
        skip_unreadable: bool = True,
        scan_desc: str = "Scanning videos",
    ):
        self.fps = fps
        self.max_frames = max_frames

        if skip_unreadable:
            self.examples: List[dict] = []
            n_drop = 0
            for s in tqdm(samples, desc=scan_desc, file=sys.stdout):
                if _is_video_readable(s["video_path"]):
                    self.examples.append(s)
                else:
                    n_drop += 1
            if n_drop:
                log.warning(
                    "MultiVideoDataset: dropped %d / %d unreadable videos",
                    n_drop, len(samples),
                )
        else:
            self.examples = list(samples)

        n_cached = sum(
            1 for s in self.examples
            if _per_frame_cache_path(s, fps, max_frames).exists()
        )
        log.info(
            "MultiVideoDataset: %d examples loaded — %d cached, %d will be extracted online",
            len(self.examples), n_cached, len(self.examples) - n_cached,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        s = self.examples[idx]

        # Sample frames at target fps up to max_frames.
        pil_images = _sample_frames(s["video_path"], self.fps, self.max_frames)

        # Pose features: load from cache if present, else extract online.
        # Both paths produce (N_frames, 1659) aligned with `pil_images`.
        cache_path = _per_frame_cache_path(s, self.fps, self.max_frames)
        if cache_path.exists():
            pose_feat = torch.load(cache_path, map_location="cpu", weights_only=False).float()
            # Defensive: if the cache was produced for a different N (e.g. fps
            # or max_frames was bumped), fall through to online extraction.
            if pose_feat.shape[0] != len(pil_images):
                pose_feat = _extract_pose_for_frames(pil_images)
        else:
            pose_feat = _extract_pose_for_frames(pil_images)

        return {
            "pil_images": pil_images,   # used only for n_frames; not sent to model
            "pose_feat":  pose_feat,
            "question":   s["question"],
            "answer":     s["answer"],
            "dataset":    s["dataset"],
            "task_type":  s["task_type"],
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_frames(video_path: str, fps: float, max_frames: int) -> List[Image.Image]:
    """Sample frames at `fps` frames-per-second, capped at `max_frames`.
    Returns a variable-length list of PIL RGB images (at least 1)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [Image.new("RGB", (224, 224))]

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1.0, video_fps / fps)

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

    return [collected.get(i, Image.new("RGB", (224, 224))) for i in indices]


# ── Collate ──────────────────────────────────────────────────────────────────

def _pad_pose_feat(batch: List[dict]) -> tuple[torch.Tensor, List[int]]:
    """Stack a batch of (N_i, feat_dim) pose tensors into (B, N_max, feat_dim)
    with zero-padding. Returns the padded tensor and per-sample frame counts."""
    feats = [ex["pose_feat"] for ex in batch]
    n_per = [f.shape[0] for f in feats]
    n_max = max(n_per)
    feat_dim = feats[0].shape[-1]
    out = torch.zeros(len(feats), n_max, feat_dim, dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out, n_per


def _collate(batch: List[dict], processor, tokenizer, pose_tokens_per_frame: int) -> dict:
    """Text + pose only — no image content is added to the chat template, and
    no `images=` is passed to the processor, so the tokenized sequence has
    zero visual tokens. Pose placeholders are inserted at the prompt/answer
    boundary; a forward pre-hook on language_model replaces them with the
    projector output during the prefill forward.
    """
    full_texts, prompt_texts = [], []

    for ex in batch:
        # Text-only content — no {"type": "image"} entries.
        content = [{"type": "text", "text": ex["question"]}]
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

    # Per-sample answer token lengths
    answer_lens = []
    for full_t, prompt_t in zip(full_texts, prompt_texts):
        answer_text = full_t[len(prompt_t):]
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        answer_lens.append(len(answer_ids))

    # Text-only tokenization (no images=).
    inputs = processor(text=full_texts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, orig_len = input_ids.shape

    # Pose-block length per sample = N_frames_i * pose_tokens_per_frame
    pose_feat_padded, n_frames_per = _pad_pose_feat(batch)
    n_pose_per = [n * pose_tokens_per_frame for n in n_frames_per]
    n_pose_max = max(n_pose_per)

    first_ans_positions = []
    for i, ans_len in enumerate(answer_lens):
        nonpad = (input_ids[i] != pad_id).sum().item()
        first_ans_positions.append(max(1, nonpad - ans_len))

    new_len = orig_len + n_pose_max
    new_ids  = torch.full((B, new_len), pad_id, dtype=input_ids.dtype)
    new_lbl  = torch.full((B, new_len), -100,   dtype=input_ids.dtype)
    new_mask = torch.zeros((B, new_len), dtype=inputs["attention_mask"].dtype)

    pose_positions = []
    for i in range(B):
        ap = first_ans_positions[i]
        n_pose_i = n_pose_per[i]
        ps, pe = ap, ap + n_pose_i

        # Prompt portion
        new_ids[i, :ap]  = input_ids[i, :ap]
        new_mask[i, :ap] = inputs["attention_mask"][i, :ap]

        # Pose placeholders (replaced by hook)
        new_ids[i, ps:pe]  = pad_id
        new_mask[i, ps:pe] = 1

        # Answer portion (shifted by n_pose_i)
        remaining = orig_len - ap
        new_ids[i, pe:pe + remaining]  = input_ids[i, ap:orig_len]
        new_mask[i, pe:pe + remaining] = inputs["attention_mask"][i, ap:orig_len]

        # Labels: only answer tokens
        nonpad = (input_ids[i] != pad_id).sum().item()
        for j in range(answer_lens[i]):
            src = nonpad - answer_lens[i] + j
            dst = pe + (src - ap)
            if 0 <= src < orig_len and 0 <= dst < new_len:
                new_lbl[i, dst] = input_ids[i, src]

        pose_positions.append((ps, pe))

    return {
        "input_ids":      new_ids,
        "attention_mask": new_mask,
        "labels":         new_lbl,
        "pose_positions": pose_positions,
        "pose_feat":      pose_feat_padded,    # (B, N_max, feat_dim)
        "n_frames_per":   n_frames_per,
    }


def _collate_gen(batch: List[dict], processor, tokenizer, pose_tokens_per_frame: int) -> dict:
    """Prompt-only inputs with per-frame pose placeholders appended at the end
    (right before where generation would start). Text-only — no images, no
    visual tokens."""
    texts = []
    for ex in batch:
        content = [{"type": "text", "text": ex["question"]}]
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )

    inputs = processor(text=texts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, orig_len = input_ids.shape

    pose_feat_padded, n_frames_per = _pad_pose_feat(batch)
    n_pose_per = [n * pose_tokens_per_frame for n in n_frames_per]
    n_pose_max = max(n_pose_per)
    new_len = orig_len + n_pose_max

    new_ids  = torch.full((B, new_len), pad_id, dtype=input_ids.dtype)
    new_mask = torch.zeros((B, new_len), dtype=inputs["attention_mask"].dtype)

    pose_positions = []
    for i in range(B):
        nonpad = (input_ids[i] != pad_id).sum().item()
        ap = nonpad
        n_pose_i = n_pose_per[i]
        ps, pe = ap, ap + n_pose_i

        new_ids[i, :ap]    = input_ids[i, :ap]
        new_mask[i, :ap]   = inputs["attention_mask"][i, :ap]

        new_ids[i, ps:pe]  = pad_id
        new_mask[i, ps:pe] = 1

        rem = orig_len - ap
        if rem > 0:
            new_ids[i, pe:pe + rem]  = input_ids[i, ap:orig_len]
            new_mask[i, pe:pe + rem] = inputs["attention_mask"][i, ap:orig_len]

        pose_positions.append((ps, pe))

    return {
        "input_ids":      new_ids,
        "attention_mask": new_mask,
        "pose_positions": pose_positions,
        "answers":        [ex["answer"]    for ex in batch],
        "datasets":       [ex["dataset"]   for ex in batch],
        "task_types":     [ex["task_type"] for ex in batch],
        "pose_feat":      pose_feat_padded,
        "n_frames_per":   n_frames_per,
    }


# ── Pose injection hook (same as train.py) ───────────────────────────────────

def _make_pose_hook(pose_embeds: torch.Tensor, pose_positions: List[tuple]):
    """Forward pre-hook for language_model: replaces pose placeholder
    embeddings with projected pose features.

    pose_embeds: (B, N_max, tokens_per_frame, H) — flattened in time-major
    order before splice.
    """
    state = {"applied": False}

    def hook_fn(module, args, kwargs):
        if state["applied"]:
            return
        ie = kwargs.get("inputs_embeds")
        if ie is None:
            return
        new_ie = ie.clone()
        pe = pose_embeds.to(device=new_ie.device, dtype=new_ie.dtype)
        if pe.dim() == 4:
            B, N_max, K, H = pe.shape
            pe_flat = pe.reshape(B, N_max * K, H)
        else:
            pe_flat = pe
        for i, (ps, pe_end) in enumerate(pose_positions):
            length = pe_end - ps
            if pe_end <= new_ie.shape[1] and length > 0:
                new_ie[i, ps:pe_end] = pe_flat[i, :length]
        state["applied"] = True
        return args, {**kwargs, "inputs_embeds": new_ie}

    return hook_fn


def _get_language_model(model, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    return unwrapped.model.language_model


# ── Generative validation ────────────────────────────────────────────────────

def _stratified_subset(dataset, max_samples: int):
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
    model, projector, val_ds, processor, tokenizer, accelerator,
    pose_tokens_per_frame: int, max_new_tokens: int = 32, max_samples: int = 0,
) -> dict:
    """Generate with pose-injected prompts (no video tokens) and compute
    strict / lenient accuracy."""
    is_main = accelerator.is_main_process
    subset = _stratified_subset(val_ds, max_samples)

    collate_fn = lambda b: _collate_gen(b, processor, tokenizer, pose_tokens_per_frame)
    gen_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
    gen_loader = accelerator.prepare(gen_loader)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    lang_model = _get_language_model(model, accelerator)

    overall = {"correct": 0, "total": 0}
    overall_lenient = {"correct": 0, "total": 0}
    by_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    by_dataset_len = defaultdict(lambda: {"correct": 0, "total": 0})
    by_tasktype = defaultdict(lambda: {"correct": 0, "total": 0})
    by_tasktype_len = defaultdict(lambda: {"correct": 0, "total": 0})

    was_training_m = model.training
    was_training_p = projector.training
    model.eval()
    projector.eval()

    with torch.no_grad():
        for batch in tqdm(gen_loader, desc="Gen-val", leave=False, disable=not is_main):
            answers        = batch.pop("answers")
            datasets       = batch.pop("datasets")
            task_types     = batch.pop("task_types")
            pose_feat      = batch.pop("pose_feat").to(dtype=torch.bfloat16)
            pose_positions = batch.pop("pose_positions")
            batch.pop("n_frames_per", None)

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

    if was_training_m:
        model.train()
    if was_training_p:
        projector.train()

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


# ── Training ─────────────────────────────────────────────────────────────────

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

    resume_ckpt = Path(args.resume_ckpt)
    resume_state = None
    if (resume_ckpt / "train_state.json").exists():
        with open(resume_ckpt / "train_state.json") as f:
            resume_state = json.load(f)
        if is_main:
            log.info("Found resume checkpoint at %s (next_epoch=%d)",
                     resume_ckpt, resume_state["next_epoch"])

    # ── Load fine-tuned backbone + projector from --pose_ckpt ───────────────
    pose_ckpt = Path(args.pose_ckpt)
    model_dir = pose_ckpt / "model"
    proj_path = pose_ckpt / "pose_projector.pt"
    cfg_path  = pose_ckpt / "config.json"

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Expected fine-tuned model at {model_dir} (produced by ../train.py)."
        )
    if not proj_path.exists():
        raise FileNotFoundError(f"Expected pose projector weights at {proj_path}.")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected config.json at {cfg_path}.")

    with open(cfg_path) as f:
        ckpt_cfg = json.load(f)
    feat_dim    = ckpt_cfg["pose_feature_dim"]
    hidden_size = ckpt_cfg["hidden_size"]
    # Infer tokens_per_frame from the saved projector tensor itself — the
    # config.json from older runs may not record it, and the CLI default can
    # disagree with what's on disk. proj.2 is Linear(hidden_size,
    # tokens_per_frame * hidden_size), so its out_features // hidden_size
    # gives the true tokens_per_frame.
    proj_sd = torch.load(proj_path, map_location="cpu", weights_only=True)
    inferred_tpf = proj_sd["proj.2.weight"].shape[0] // hidden_size
    cfg_tpf = ckpt_cfg.get("pose_tokens_per_frame")
    if cfg_tpf is not None and int(cfg_tpf) != inferred_tpf:
        log.warning(
            "config.json says pose_tokens_per_frame=%s but saved projector "
            "implies %d; using %d (state_dict wins).",
            cfg_tpf, inferred_tpf, inferred_tpf,
        )
    tokens_per_frame = inferred_tpf

    log.info("Loading fine-tuned backbone from: %s", model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer = processor.tokenizer

    assert hasattr(model, "model"),   "Expected model.model (backbone)"
    assert hasattr(model, "lm_head"), "Expected model.lm_head"

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        log.info("Backbone frozen — training projector only.")
    else:
        for p in model.parameters():
            p.requires_grad = True
        model.train()

    if args.gradient_checkpointing and not args.freeze_backbone:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Backbone params: %s  trainable: %s",
             f"{total_params:,}", f"{trainable:,}")

    # Construct projector with the SAME tokens_per_frame as the saved one.
    projector = PoseFeatureProjector(
        feat_dim, hidden_size, tokens_per_frame=tokens_per_frame,
    ).to(dtype=torch.bfloat16)
    projector.load_state_dict(proj_sd)
    for p in projector.parameters():
        p.requires_grad = True
    log.info(
        "Loaded pose projector from %s  (%d → %d × %d tokens/frame, %s params)",
        proj_path, feat_dim, hidden_size, tokens_per_frame,
        f"{sum(p.numel() for p in projector.parameters()):,}",
    )

    # ── Datasets (offline cache if present, else online; pixels never reach the model) ──
    splits = get_all_samples()
    train_ds = MultiVideoDataset(
        splits["train"], fps=args.fps, max_frames=args.max_frames,
        scan_desc="Scanning train videos",
    )
    val_ds = MultiVideoDataset(
        splits["val"], fps=args.fps, max_frames=args.max_frames,
        scan_desc="Scanning val videos",
    )

    collate_fn = lambda b: _collate(b, processor, tokenizer, tokens_per_frame)

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

    if is_main:
        log.info("accelerator.prepare(model, projector, loaders) — wrapping with FSDP, this can take a minute on a 9B model …")
    model, projector, train_loader, val_loader = accelerator.prepare(
        model, projector, train_loader, val_loader
    )
    if is_main:
        log.info("accelerator.prepare done.")

    if args.torch_compile:
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)
        projector = torch.compile(projector)

    # Two optimizers: 1:1 mapping with each FSDP module is required for save.
    if args.freeze_backbone:
        model_optimizer = None
    else:
        model_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            foreach=False,
            fused=False,
        )
    proj_optimizer = torch.optim.AdamW(
        list(projector.parameters()),
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
        # Clamp so cos doesn't wrap past π and send the LR back up to peak.
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    proj_scheduler = torch.optim.lr_scheduler.LambdaLR(proj_optimizer, lr_lambda)
    if model_optimizer is not None:
        model_scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda)
        model_optimizer, proj_optimizer, model_scheduler, proj_scheduler = accelerator.prepare(
            model_optimizer, proj_optimizer, model_scheduler, proj_scheduler
        )
    else:
        model_scheduler = None
        proj_optimizer, proj_scheduler = accelerator.prepare(proj_optimizer, proj_scheduler)

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

    if model_optimizer is not None:
        model_optimizer.zero_grad()
    proj_optimizer.zero_grad()

    lang_model = _get_language_model(model, accelerator)

    if is_main:
        log.info("Initializing FSDP …")
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        model(input_ids=_dummy, attention_mask=torch.ones_like(_dummy))
    accelerator.wait_for_everyone()

    # Latest gen-val accuracies, surfaced in the tqdm postfix between val runs.
    latest_val = {"strict": None, "lenient": None}

    if not resume_state and not args.skip_baseline_eval:
        if is_main:
            log.info(
                "Fresh run — evaluating no-finetune baseline "
                "(loaded train.py checkpoint, video tokens removed) …"
            )

        baseline_results = _run_generative_validation(
            model, projector, val_ds, processor, tokenizer, accelerator,
            pose_tokens_per_frame=tokens_per_frame,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=args.val_max_samples,
        )
        latest_val["strict"]  = baseline_results["overall_strict"]
        latest_val["lenient"] = baseline_results["overall_lenient"]
        if is_main:
            _log_gen_results(
                "Baseline (loaded train.py ckpt, no video, no finetune)",
                baseline_results,
            )
            if use_wandb:
                wandb.log({
                    "baseline/strict":  baseline_results["overall_strict"],
                    "baseline/lenient": baseline_results["overall_lenient"],
                }, step=0)

    for epoch in range(start_epoch, args.num_epochs + 1):
        if args.freeze_backbone:
            model.eval()
        else:
            model.train()
        projector.train()

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
            with accelerator.accumulate(model, projector):
                pose_feat      = batch.pop("pose_feat").to(dtype=torch.bfloat16)
                labels         = batch.pop("labels")
                pose_positions = batch.pop("pose_positions")
                batch.pop("n_frames_per", None)

                pose_embeds = projector(pose_feat)
                hook_fn = _make_pose_hook(pose_embeds, pose_positions)
                handle = lang_model.register_forward_pre_hook(hook_fn, with_kwargs=True)

                # Text + pose only — no pixel_values / grid_thw / mm_token_type_ids.
                backbone_out = model.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
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
                    params_to_clip = list(projector.parameters())
                    if model_optimizer is not None:
                        params_to_clip = list(model.parameters()) + params_to_clip
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if model_optimizer is not None:
                    model_optimizer.step()
                    model_scheduler.step()
                    model_optimizer.zero_grad()
                proj_optimizer.step()
                proj_scheduler.step()
                proj_optimizer.zero_grad()

            running_loss += loss.item()
            if is_main:
                postfix = {"loss": f"{loss.item():.4f}"}
                if latest_val["strict"] is not None:
                    postfix["strict"]  = f"{latest_val['strict']:.3f}"
                    postfix["lenient"] = f"{latest_val['lenient']:.3f}"
                pbar.set_postfix(**postfix)

            if accelerator.sync_gradients:
                global_step += 1
                step_metrics = {"epoch": epoch} if use_wandb else {}

                if global_step % args.save_steps == 0:
                    torch.cuda.synchronize()
                    _save_resume_checkpoint(
                        accelerator, resume_ckpt,
                        next_epoch=epoch, global_step=global_step,
                        step_in_epoch=step + 1,
                        best_val_loss=best_val_loss, feat_dim=feat_dim, hidden_size=hidden_size,
                        tokens_per_frame=tokens_per_frame,
                    )

                if args.val_steps > 0 and global_step % args.val_steps == 0:
                    if is_main:
                        log.info("Running generative validation at step %d …", global_step)
                    gen_results = _run_generative_validation(
                        model, projector, val_ds, processor, tokenizer, accelerator,
                        pose_tokens_per_frame=tokens_per_frame,
                        max_new_tokens=args.val_max_new_tokens,
                        max_samples=args.val_max_samples,
                    )
                    latest_val["strict"]  = gen_results["overall_strict"]
                    latest_val["lenient"] = gen_results["overall_lenient"]
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

                    backbone_lr = (model_scheduler.get_last_lr()[0]
                                   if model_scheduler is not None else 0.0)
                    proj_lr     = proj_scheduler.get_last_lr()[0]

                    log.info(
                        "epoch %d  step %d  loss %.4f  backbone_lr %.2e  proj_lr %.2e",
                        epoch, global_step, avg_loss, backbone_lr, proj_lr,
                    )
                    postfix = {"loss": f"{avg_loss:.4f}"}
                    if latest_val["strict"] is not None:
                        postfix["strict"]  = f"{latest_val['strict']:.3f}"
                        postfix["lenient"] = f"{latest_val['lenient']:.3f}"
                    pbar.set_postfix(**postfix)
                    step_metrics.update({
                        "train/loss":        avg_loss,
                        "train/backbone_lr": backbone_lr,
                        "train/proj_lr":     proj_lr,
                    })

                if use_wandb and step_metrics and is_main:
                    wandb.log(step_metrics, step=global_step)

        # ── Validation (loss) ─────────────────────────────────────────────
        model.eval()
        projector.eval()
        val_loss_sum = torch.zeros(1, device=device)
        n_batches    = torch.zeros(1, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False,
                              disable=not is_main):
                pose_feat      = batch.pop("pose_feat").to(dtype=torch.bfloat16)
                labels         = batch.pop("labels")
                pose_positions = batch.pop("pose_positions")
                batch.pop("n_frames_per", None)

                pose_embeds = projector(pose_feat)
                hook_fn = _make_pose_hook(pose_embeds, pose_positions)
                handle = lang_model.register_forward_pre_hook(hook_fn, with_kwargs=True)

                backbone_out = model.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
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

        if is_main:
            log.info("Running end-of-epoch generative validation …")
        gen_results = _run_generative_validation(
            model, projector, val_ds, processor, tokenizer, accelerator,
            pose_tokens_per_frame=tokens_per_frame,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=0,
        )
        latest_val["strict"]  = gen_results["overall_strict"]
        latest_val["lenient"] = gen_results["overall_lenient"]
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

        ckpt_dir = output_dir / f"epoch-{epoch}"
        if is_main:
            ckpt_dir.mkdir(exist_ok=True)
        accelerator.wait_for_everyone()

        proj_state = accelerator.get_state_dict(projector)
        if is_main:
            torch.save(proj_state, ckpt_dir / "pose_projector.pt")
            _save_config(ckpt_dir, feat_dim, hidden_size, tokens_per_frame, args)

        new_best = val_loss < best_val_loss
        if new_best:
            best_val_loss = val_loss
            best_dir = output_dir / "best"
            if is_main:
                best_dir.mkdir(exist_ok=True)
                torch.save(proj_state, best_dir / "pose_projector.pt")
                _save_config(best_dir, feat_dim, hidden_size, tokens_per_frame, args)
            accelerator.wait_for_everyone()

            if args.save_full_model and not args.freeze_backbone:
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
            best_val_loss=best_val_loss, feat_dim=feat_dim, hidden_size=hidden_size,
            tokens_per_frame=tokens_per_frame,
        )

    if is_main:
        log.info("Training complete. Best val_loss: %.4f", best_val_loss)
        if use_wandb:
            wandb.finish()


def _save_config(directory: Path, feat_dim: int, hidden_size: int,
                 tokens_per_frame: int, args) -> None:
    cfg = {
        "pose_feature_dim":      feat_dim,
        "hidden_size":           hidden_size,
        "pose_tokens_per_frame": tokens_per_frame,
        "pose_ckpt":             args.pose_ckpt,
        "backbone_lr":           args.backbone_lr,
        "lr":                    args.lr,
        "freeze_backbone":       args.freeze_backbone,
        "no_video":              True,
    }
    with open(directory / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def _reset_fsdp_training_state(accelerator) -> None:
    """Workaround for PyTorch FSDP bug: after backward(), handle._training_state
    stays at BACKWARD_POST and only resets on the next forward. Calling
    state_dict() in between trips an assertion. Force every FSDP module / handle
    back to IDLE so save_state can unshard cleanly."""
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
    feat_dim: int,
    hidden_size: int,
    tokens_per_frame: int = 16,
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
            "next_epoch":            next_epoch,
            "global_step":           global_step,
            "step_in_epoch":         step_in_epoch,
            "best_val_loss":         best_val_loss,
            "feat_dim":              feat_dim,
            "hidden_size":           hidden_size,
            "pose_tokens_per_frame": tokens_per_frame,
        }
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(state, f, indent=2)
        log.info("Resume checkpoint saved (next_epoch=%d, global_step=%d)",
                 next_epoch, global_step)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Continue fine-tuning train.py's pose checkpoint with no video tokens"
    )
    # Model / output
    p.add_argument("--pose_ckpt",
                   default="/orcd/compute/ppliang/001/qwen_multi/best",
                   help="Directory produced by ../train.py — must contain "
                        "model/, pose_projector.pt, and config.json.")
    p.add_argument("--output_dir",   default="/orcd/compute/ppliang/001/qwen_pose_only")
    p.add_argument("--resume_ckpt",  default="/orcd/compute/ppliang/001/qwen_pose_only/resume_ckpt")

    # Data — must match train.py's video sampling so per-frame pose counts
    # align with what the saved projector was trained on.
    p.add_argument("--pose_tokens_per_frame", type=int, default=16,
                   help="Used only as a fallback when the saved checkpoint's "
                        "config.json lacks the field. Otherwise the value is "
                        "read from the checkpoint to match the saved projector.")
    p.add_argument("--fps",        type=float, default=1.0,
                   help="Target sampling rate (frames/sec) for video frames "
                        "fed to the pose extractor. Match train.py's value.")
    p.add_argument("--max_frames", type=int,   default=8,
                   help="Maximum number of video frames per clip. Match train.py.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--num_workers", type=int,  default=4,
                   help="DataLoader worker processes (decode video + extract pose).")

    # Training
    p.add_argument("--num_epochs",   type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--backbone_lr",  type=float, default=1e-5,
                   help="LR for the Qwen backbone (continues fine-tuning the "
                        "train.py checkpoint with no video tokens).")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="LR for pose projector parameters.")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--freeze_backbone", action="store_true", default=False,
                   help="Freeze the Qwen backbone — train projector only. "
                        "Cleaner ablation of 'can the projector bridge the "
                        "missing-video gap?'  Default: train both.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--torch_compile", action="store_true", default=False)
    p.add_argument("--no_torch_compile", dest="torch_compile", action="store_false")
    p.add_argument("--save_full_model", action="store_true", default=True,
                   help="Save full Qwen model at best checkpoint (~14 GB).")
    p.add_argument("--save_steps",     type=int, default=10)
    p.add_argument("--val_steps",      type=int, default=10,
                   help="Run generative validation every N steps (0 = epoch-end only).")
    p.add_argument("--val_max_samples", type=int, default=200,
                   help="Max samples for mid-training generative validation (0 = all).")
    p.add_argument("--val_max_new_tokens", type=int, default=32)
    p.add_argument("--skip_baseline_eval", action="store_true", default=False,
                   help="Skip the baseline generative validation pass before "
                        "the first training step (saves several minutes of "
                        "video decode + MediaPipe + generation).")

    # Logging
    p.add_argument("--log_steps",       type=int,  default=10)
    p.add_argument("--wandb_project",   type=str,  default="qwen-pose-only")
    p.add_argument("--wandb_run_name",  type=str,  default=None)
    p.add_argument("--no_wandb",        action="store_true")
    p.add_argument("--log_dir",         type=str, default=None,
                   help="Directory for run log files. Defaults to <output_dir>/logs.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
