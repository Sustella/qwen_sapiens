#!/usr/bin/env python3
"""
train.py — Fine-tune Qwen3.5-VL on the default multi-dataset mix
(QVID + Kinetics-400 + HMDB51 + ShareGPT4Video) with pose features.

Pose features are projected into the LLM embedding space and injected into the
input sequence (between the prompt and answer tokens) so the transformer attends
to them during generation.  Loss is standard causal-LM cross-entropy on all
answer tokens, conditioned on [video + text + pose].

Usage
-----
  python train.py \\
      --model_name Qwen/Qwen3.5-VL-7B-Instruct \\
      --output_dir /orcd/compute/ppliang/001/qwen_multi_new
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

from multi_dataset import get_all_samples

# Path setup so utils/pose_features.py is importable here.
_REPO_DIR = Path(__file__).resolve().parent
if str(_REPO_DIR / "utils") not in sys.path:
    sys.path.insert(0, str(_REPO_DIR / "utils"))
from pose_features import LandmarkerManager, TOTAL_DIM as POSE_TOTAL_DIM  # noqa: E402

# ── Online pose extractor ────────────────────────────────────────────────────
# Mirrors the offline `extract_features_for_frames` from utils/pose_features.py:
# fresh `LandmarkerManager(running_mode='video')` per video, monotonically
# increasing 33ms timestamps, same feature concatenation order. The only
# difference vs offline is *which* frames are fed in — here it's the same
# PIL frames the vision encoder will see, processed in temporal order.
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


# ── Pose projector ────────────────────────────────────────────────────────────

class PoseFeatureProjector(nn.Module):
    """Per-frame pose projector.

    Takes (..., feature_dim) per-frame pose features and produces
    `tokens_per_frame` tokens of width `hidden_size` for each frame.

    Forward input  : (B, N_frames, feature_dim)
    Forward output : (B, N_frames, tokens_per_frame, hidden_size)

    The intermediate hidden width matches `hidden_size`. The output head
    Linear maps to `tokens_per_frame * hidden_size` and is reshaped.
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
        # x : (..., feature_dim) → (..., tokens_per_frame, hidden_size)
        out = self.proj(x)
        return out.view(*x.shape[:-1], self.tokens_per_frame, self.hidden_size)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiVideoDataset(Dataset):
    """
    Unified dataset over QVID, Kinetics-400, and HMDB51.
    Each sample is a (video_frames, pose_features, question, answer) tuple.

    Pose features may come from either source, transparently per-sample:
      • offline cache file at ``_per_frame_cache_path(s, fps, max_frames)``
        (produced by ``extract_poses_per_frame.py``) — loaded directly.
      • otherwise extracted online in the DataLoader worker via MediaPipe
        on the same PIL frames the vision encoder sees.

    Either way the resulting tensor has shape ``(N_frames, 1659)`` aligned
    1-to-1 with the frames in ``pil_images``, and the projector emits
    ``tokens_per_frame`` tokens per video frame.

    Unreadable videos are dropped at init via a one-time pre-scan with a
    progress bar (mirrors the old ``skip_missing_pose=True`` behavior, which
    excluded samples whose offline ``.pt`` failed to extract).
    """

    def __init__(
        self,
        samples: List[dict],
        fps: float = 2.0,
        max_frames: int = 32,
        # Back-compat shim: callers may still pass `pose_sample_n=`; it is
        # ignored under the per-frame regime (pose count is now derived
        # from the number of sampled video frames). The number of *tokens
        # per pose frame* is decided downstream by the projector.
        pose_sample_n: int = None,
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


# ── Collate ───────────────────────────────────────────────────────────────────

def _pad_pose_feat(batch: List[dict]) -> tuple[torch.Tensor, List[int]]:
    """Stack a batch of (N_i, feat_dim) pose tensors into (B, N_max, feat_dim)
    with zero-padding. Returns the padded tensor and per-sample frame counts."""
    feats = [ex["pose_feat"] for ex in batch]                    # each (N_i, D)
    n_per = [f.shape[0] for f in feats]
    n_max = max(n_per)
    feat_dim = feats[0].shape[-1]
    out = torch.zeros(len(feats), n_max, feat_dim, dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out, n_per


def _collate(batch: List[dict], processor, tokenizer, pose_tokens_per_frame: int) -> dict:
    """
    Build processor inputs for a batch, inserting per-frame pose placeholder
    tokens at the prompt/answer boundary. Each sampled video frame contributes
    ``pose_tokens_per_frame`` tokens, so the pose-block length per sample is
    ``N_frames_i * pose_tokens_per_frame``. Across a batch this varies if the
    samples have different frame counts; the sequence is right-padded to the
    batch max via ``pad_id`` placeholders that are masked out.

    Labels are -100 for prompt and pose positions; only answer tokens have real
    labels so the loss is on answer tokens conditioned on [prompt + pose].
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

    # Per-sample answer token lengths (text-only; no visual tokens in answer)
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

    B, orig_len = input_ids.shape

    # Pose-block length per sample = N_frames_i * pose_tokens_per_frame
    pose_feat_padded, n_frames_per = _pad_pose_feat(batch)
    n_pose_per = [n * pose_tokens_per_frame for n in n_frames_per]
    n_pose_max = max(n_pose_per)

    first_ans_positions = []
    for i, ans_len in enumerate(answer_lens):
        nonpad = (input_ids[i] != pad_id).sum().item()
        first_ans_positions.append(max(1, nonpad - ans_len))

    # New seq length = orig_len + n_pose_max (right-pad shorter pose blocks)
    new_len = orig_len + n_pose_max
    new_ids  = torch.full((B, new_len), pad_id, dtype=input_ids.dtype)
    new_lbl  = torch.full((B, new_len), -100,   dtype=input_ids.dtype)
    new_mask = torch.zeros((B, new_len), dtype=inputs["attention_mask"].dtype)
    has_mm   = "mm_token_type_ids" in inputs
    new_mm   = torch.zeros((B, new_len), dtype=inputs["mm_token_type_ids"].dtype) if has_mm else None

    pose_positions = []  # (start, end) per sample — used by the hook
    for i in range(B):
        ap = first_ans_positions[i]
        n_pose_i = n_pose_per[i]
        ps, pe = ap, ap + n_pose_i

        # Prompt portion  [0 .. ap)
        new_ids[i, :ap]  = input_ids[i, :ap]
        new_mask[i, :ap] = inputs["attention_mask"][i, :ap]
        if has_mm:
            new_mm[i, :ap] = inputs["mm_token_type_ids"][i, :ap]

        # Pose placeholders  [ap .. ap+n_pose_i)
        new_ids[i, ps:pe]  = pad_id   # placeholder; replaced by hook
        new_mask[i, ps:pe] = 1

        # Answer portion  [ap+n_pose_i ..)
        remaining = orig_len - ap
        new_ids[i, pe:pe + remaining]  = input_ids[i, ap:orig_len]
        new_mask[i, pe:pe + remaining] = inputs["attention_mask"][i, ap:orig_len]
        if has_mm:
            new_mm[i, pe:pe + remaining] = inputs["mm_token_type_ids"][i, ap:orig_len]

        # Labels: only answer tokens (shifted positions)
        nonpad = (input_ids[i] != pad_id).sum().item()
        for j in range(answer_lens[i]):
            src = nonpad - answer_lens[i] + j
            dst = pe + (src - ap)
            if 0 <= src < orig_len and 0 <= dst < new_len:
                new_lbl[i, dst] = input_ids[i, src]

        pose_positions.append((ps, pe))

    result = {
        "input_ids":      new_ids,
        "attention_mask":  new_mask,
        "labels":          new_lbl,
        "pose_positions":  pose_positions,
        "pose_feat":       pose_feat_padded,    # (B, N_max, feat_dim)
        "n_frames_per":    n_frames_per,        # used by the projection hook
    }
    if has_mm:
        result["mm_token_type_ids"] = new_mm
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in inputs:
            result[k] = inputs[k]
    return result


def _collate_gen(batch: List[dict], processor, tokenizer, pose_tokens_per_frame: int) -> dict:
    """Build prompt-only inputs with per-frame pose placeholders appended at
    the end (right before where generation would start). Same per-frame pose
    semantics as ``_collate``."""
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
    input_ids = inputs["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, orig_len = input_ids.shape

    pose_feat_padded, n_frames_per = _pad_pose_feat(batch)
    n_pose_per = [n * pose_tokens_per_frame for n in n_frames_per]
    n_pose_max = max(n_pose_per)
    new_len = orig_len + n_pose_max

    new_ids  = torch.full((B, new_len), pad_id, dtype=input_ids.dtype)
    new_mask = torch.zeros((B, new_len), dtype=inputs["attention_mask"].dtype)
    has_mm   = "mm_token_type_ids" in inputs
    new_mm   = torch.zeros((B, new_len), dtype=inputs["mm_token_type_ids"].dtype) if has_mm else None

    pose_positions = []
    for i in range(B):
        nonpad = (input_ids[i] != pad_id).sum().item()
        ap = nonpad  # insert at end of prompt (all tokens are prompt)
        n_pose_i = n_pose_per[i]
        ps, pe = ap, ap + n_pose_i

        new_ids[i, :ap]    = input_ids[i, :ap]
        new_mask[i, :ap]   = inputs["attention_mask"][i, :ap]
        if has_mm:
            new_mm[i, :ap] = inputs["mm_token_type_ids"][i, :ap]

        new_ids[i, ps:pe]  = pad_id
        new_mask[i, ps:pe] = 1

        rem = orig_len - ap
        if rem > 0:
            new_ids[i, pe:pe + rem]  = input_ids[i, ap:orig_len]
            new_mask[i, pe:pe + rem] = inputs["attention_mask"][i, ap:orig_len]
            if has_mm:
                new_mm[i, pe:pe + rem] = inputs["mm_token_type_ids"][i, ap:orig_len]

        pose_positions.append((ps, pe))

    result = {
        "input_ids":       new_ids,
        "attention_mask":   new_mask,
        "pose_positions":   pose_positions,
        "answers":         [ex["answer"]    for ex in batch],
        "datasets":        [ex["dataset"]   for ex in batch],
        "task_types":      [ex["task_type"] for ex in batch],
        "pose_feat":       pose_feat_padded,
        "n_frames_per":    n_frames_per,
    }
    if has_mm:
        result["mm_token_type_ids"] = new_mm
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in inputs:
            result[k] = inputs[k]
    return result


# ── Pose injection hook ──────────────────────────────────────────────────────

def _make_pose_hook(pose_embeds: torch.Tensor, pose_positions: List[tuple]):
    """Return a forward pre-hook for the language_model that replaces pose
    placeholder embeddings with projected pose features.

    ``pose_embeds``: (B, N_max, tokens_per_frame, H) from the projector. Each
    sample's pose block is the per-frame tokens flattened in time-major order:
    [frame_0 token_0, frame_0 token_1, …, frame_(N-1) token_(K-1)].
    ``pose_positions``: list of (start, end) per batch element. ``end - start``
    equals ``n_frames_i * tokens_per_frame`` and may differ across the batch.

    The hook fires once (on the prefill forward) then becomes a no-op so
    that subsequent generate() steps are unaffected.
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
    """Navigate through FSDP wrapping to get the language_model sub-module
    for hook registration."""
    unwrapped = accelerator.unwrap_model(model)
    return unwrapped.model.language_model


# ── Generative validation ────────────────────────────────────────────────────

def _stratified_subset(dataset, max_samples: int):
    """Return a Subset that samples proportionally from each dataset
    (identified by the 'dataset' field), so all datasets are represented."""
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    # Group indices by dataset name
    by_ds = defaultdict(list)
    for idx in range(len(dataset)):
        ex = dataset[idx] if not isinstance(dataset, torch.utils.data.Subset) else dataset.dataset[dataset.indices[idx]]
        by_ds[ex["dataset"]].append(idx)

    # Proportional allocation (at least 1 per dataset if available)
    total = sum(len(v) for v in by_ds.values())
    selected = []
    for ds_name, indices in sorted(by_ds.items()):
        n = max(1, round(max_samples * len(indices) / total))
        # Evenly spaced to get a representative slice
        step = max(1, len(indices) // n)
        selected.extend(indices[::step][:n])

    # Trim to target
    selected = selected[:max_samples]
    return torch.utils.data.Subset(dataset, selected)


def _run_generative_validation(
    model, projector, val_ds, processor, tokenizer, accelerator,
    pose_tokens_per_frame: int, max_new_tokens: int = 32, max_samples: int = 0,
) -> dict:
    """Generate with pose-injected prompts and compute strict/lenient accuracy."""
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
        gen_bar = tqdm(gen_loader, desc="Gen-val", leave=False,
                       disable=not is_main, file=sys.stdout)
        gen_bar.refresh()
        if is_main:
            print(f"[gen-val] starting, total={len(gen_loader)}", flush=True)
        for _gv_idx, batch in enumerate(gen_bar):
            if is_main:
                print(f"[gen-val] batch {_gv_idx + 1}/{len(gen_loader)}", flush=True)
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


def _run_generative_validation_pretrained(
    model, val_ds, processor, tokenizer, accelerator,
    max_new_tokens: int = 32, max_samples: int = 0,
) -> dict:
    """Generate WITHOUT the projector (plain model.generate).
    Used for pretrained baseline measurement.  No pose tokens are inserted."""
    is_main = accelerator.is_main_process

    subset = _stratified_subset(val_ds, max_samples)

    # Use a vanilla collate that does NOT insert pose placeholders
    def _collate_vanilla(batch):
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

    gen_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=_collate_vanilla)
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
        pre_bar = tqdm(gen_loader, desc="Pretrained-val", leave=False,
                       disable=not is_main, file=sys.stdout)
        pre_bar.refresh()
        if is_main:
            print(f"[pretrained-val] starting, total={len(gen_loader)}", flush=True)
        for _pv_idx, batch in enumerate(pre_bar):
            if is_main:
                print(f"[pretrained-val] batch {_pv_idx + 1}/{len(gen_loader)}", flush=True)
            answers    = batch.pop("answers")
            datasets   = batch.pop("datasets")
            task_types = batch.pop("task_types")

            prompt_len = batch["input_ids"].shape[1]
            gen_kwargs = {k: v for k, v in batch.items() if v is not None}
            try:
                gen_ids = model.generate(
                    **gen_kwargs, max_new_tokens=32, do_sample=False,
                    pad_token_id=pad_id, eos_token_id=eos_id,
                )
                new_ids = gen_ids[:, prompt_len:]
            except Exception as e:
                log.warning("Pretrained generation failed, skipping: %s", e)
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
    """Pretty-print generative validation results."""
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


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # ── Accelerator + FSDP plugin ─────────────────────────────────────────────
    # Runs as plain single-process when launched via `python train.py`;
    # shards model across GPUs when launched via `accelerate launch train.py`.
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,  # bf16 everywhere → grads match bf16 params + Adam state
            buffer_dtype=torch.bfloat16,
        ),
        auto_wrap_policy=functools.partial(
            size_based_auto_wrap_policy, min_num_params=int(1e8)
        ),
        use_orig_params=True,
        cpu_offload=False,
        activation_checkpointing=False,  # HF's gradient_checkpointing handles this
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
    )
    # Do NOT pass mixed_precision="bf16" here — FSDP's own MixedPrecision
    # policy already casts compute to bf16 and keeps grads/params/Adam state
    # in fp32. Stacking Accelerator autocast on top makes non-FSDP modules
    # (e.g. the projector, or backbone submodules below the auto-wrap
    # threshold) emit bf16 grads while their params/state are fp32, which
    # crashes optimizer.step with the "expected dtype float" error.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        fsdp_plugin=fsdp_plugin,
    )
    set_seed(args.seed)
    device = accelerator.device
    is_main = accelerator.is_main_process

    # ── Log file (main process only) ─────────────────────────────────────────
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

    # ── WandB (main process only) ─────────────────────────────────────────────
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

    # ── Resume checkpoint detection ───────────────────────────────────────────
    resume_ckpt = Path(args.resume_ckpt)
    resume_state = None
    if (resume_ckpt / "train_state.json").exists():
        with open(resume_ckpt / "train_state.json") as f:
            resume_state = json.load(f)
        if is_main:
            log.info("Found resume checkpoint at %s (next_epoch=%d)",
                     resume_ckpt, resume_state["next_epoch"])

    # ── Model ─────────────────────────────────────────────────────────────────
    # Always load from base model — accelerator.load_state() overlays trained
    # weights on resume. No device_map: FSDP handles placement.
    log.info("Loading model: %s", args.model_name)
    # Load in bf16 so params, grads, and Adam state all live in bf16 —
    # avoids every dtype-mismatch variant of the FSDP + AdamW bug.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    assert hasattr(model, "model"),   "Expected model.model (backbone)"
    assert hasattr(model, "lm_head"), "Expected model.lm_head"

    for p in model.parameters():
        p.requires_grad = True
    model.train()

    # Gradient checkpointing must be enabled BEFORE FSDP wrap.
    # use_reentrant=False is required for FSDP compatibility.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled.")

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Total params: %s", f"{total_params:,}")

    # ── Pose projector ────────────────────────────────────────────────────────
    # Pose features are extracted online; the dim is fixed by the MediaPipe
    # landmark counts (33+478+21+21 keypoints * xyz = 1659).
    feat_dim = (resume_state["feat_dim"] if resume_state else None) or args.pose_feature_dim
    if feat_dim is None:
        feat_dim = POSE_TOTAL_DIM
        log.info("Using pose_feature_dim = %d (MediaPipe pose+face+L_hand+R_hand)", feat_dim)

    hidden_size = model.config.get_text_config().hidden_size
    projector = PoseFeatureProjector(
        feat_dim, hidden_size, tokens_per_frame=args.pose_tokens_per_frame,
    ).to(dtype=torch.bfloat16)
    log.info(
        "PoseFeatureProjector: %d → %d × %d tokens/frame  (%s params)",
        feat_dim, hidden_size, args.pose_tokens_per_frame,
        f"{sum(p.numel() for p in projector.parameters()):,}",
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    splits = get_all_samples()
    train_ds = MultiVideoDataset(
        splits["train"], fps=args.fps, max_frames=args.max_frames,
        scan_desc="Scanning train videos",
    )
    val_ds = MultiVideoDataset(
        splits["val"], fps=args.fps, max_frames=args.max_frames,
        scan_desc="Scanning val videos",
    )

    collate_fn = lambda b: _collate(b, processor, tokenizer, args.pose_tokens_per_frame)

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

    # ── Prepare model + projector + loaders (FSDP wrap happens here) ──────────
    if is_main:
        print("[stage] accelerator.prepare(model, projector, loaders) — FSDP wrap …", flush=True)
    model, projector, train_loader, val_loader = accelerator.prepare(
        model, projector, train_loader, val_loader
    )
    if is_main:
        print("[stage] accelerator.prepare(model, …) done.", flush=True)

    # torch.compile AFTER FSDP wrap (if enabled).
    if args.torch_compile:
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)
        projector = torch.compile(projector)

    # ── Optimizers — one per FSDP-wrapped module ──────────────────────────────
    # Accelerate's FSDP optimizer save requires a 1:1 mapping between each
    # optimizer and its FSDP model (FSDP.optim_state_dict looks every param up
    # in that one model's param_to_fqns). Using a single optimizer across two
    # FSDP modules trips a KeyError on save, so we keep them separate.
    model_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        foreach=False,  # FSDP use_orig_params can mix dtypes within a group
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
        # Clamp at 1.0 so cos doesn't wrap past π back up to +1 — without this,
        # any step beyond total_steps (e.g. on a resumed run whose new
        # total_steps is smaller than the saved scheduler step) sends the LR
        # back to peak and drives loss back UP.
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    model_scheduler = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda)
    proj_scheduler = torch.optim.lr_scheduler.LambdaLR(proj_optimizer, lr_lambda)
    if is_main:
        print("[stage] accelerator.prepare(optimizers, schedulers) …", flush=True)
    model_optimizer, proj_optimizer, model_scheduler, proj_scheduler = accelerator.prepare(
        model_optimizer, proj_optimizer, model_scheduler, proj_scheduler
    )
    if is_main:
        print("[stage] accelerator.prepare(optimizers, …) done.", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
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

    model_optimizer.zero_grad()
    proj_optimizer.zero_grad()

    # ── Get language_model handle for pose injection hooks ────────────────
    lang_model = _get_language_model(model, accelerator)

    # ── Warm up FSDP ─────────────────────────────────────────────────────
    # FSDP lazy-initializes on the first forward through the ROOT wrapper.
    # model.generate() bypasses the root (HF generate() calls self(...) on
    # the unwrapped module), so we must trigger init through model() first.
    if is_main:
        log.info("Initializing FSDP …")
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        model(input_ids=_dummy, attention_mask=torch.ones_like(_dummy))
    accelerator.wait_for_everyone()

    # ── Pre-training baseline (fresh runs only) ──────────────────────────
    if not resume_state:
        if is_main:
            log.info("Fresh run — evaluating pretrained baselines on validation …")

        # (1) Pretrained model alone (no pose tokens)
        pt_results = _run_generative_validation_pretrained(
            model, val_ds, processor, tokenizer, accelerator,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=args.val_max_samples,
        )
        if is_main:
            _log_gen_results("Baseline (pretrained, no pose)", pt_results)
            if use_wandb:
                wandb.log({
                    "baseline/pretrained_strict":  pt_results["overall_strict"],
                    "baseline/pretrained_lenient": pt_results["overall_lenient"],
                }, step=0)

        # (2) Pretrained model + randomly initialized projector (with pose tokens)
        rp_results = _run_generative_validation(
            model, projector, val_ds, processor, tokenizer, accelerator,
            pose_tokens_per_frame=args.pose_tokens_per_frame,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=args.val_max_samples,
        )
        if is_main:
            _log_gen_results("Baseline (pretrained + random projector)", rp_results)
            if use_wandb:
                wandb.log({
                    "baseline/random_proj_strict":  rp_results["overall_strict"],
                    "baseline/random_proj_lenient": rp_results["overall_lenient"],
                }, step=0)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.num_epochs + 1):
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
            file=sys.stdout,
        )
        pbar.refresh()

        for local_step, batch in enumerate(pbar):
            step = local_step + step_offset
            with accelerator.accumulate(model, projector):
                pose_feat      = batch.pop("pose_feat").to(dtype=torch.bfloat16)
                labels         = batch.pop("labels")
                pose_positions = batch.pop("pose_positions")
                batch.pop("n_frames_per", None)

                # Project pose features and register hook to inject them
                pose_embeds = projector(pose_feat)
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
                        list(model.parameters()) + list(projector.parameters()),
                        args.max_grad_norm,
                    )

                model_optimizer.step()
                proj_optimizer.step()
                model_scheduler.step()
                proj_scheduler.step()
                model_optimizer.zero_grad()
                proj_optimizer.zero_grad()

            running_loss += loss.item()
            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if accelerator.sync_gradients:
                global_step += 1

                # Accumulate all wandb metrics for this step into one dict
                # to avoid multiple wandb.log() calls at the same step.
                step_metrics = {"epoch": epoch} if use_wandb else {}

                if global_step % args.save_steps == 0:
                    torch.cuda.synchronize()
                    _save_resume_checkpoint(
                        accelerator, resume_ckpt,
                        next_epoch=epoch, global_step=global_step,
                        step_in_epoch=step + 1,
                        best_val_loss=best_val_loss, feat_dim=feat_dim, hidden_size=hidden_size,
                        tokens_per_frame=args.pose_tokens_per_frame,
                    )

                if args.val_steps > 0 and global_step % args.val_steps == 0:
                    if is_main:
                        log.info("Running generative validation at step %d …", global_step)
                    gen_results = _run_generative_validation(
                        model, projector, val_ds, processor, tokenizer, accelerator,
                        pose_tokens_per_frame=args.pose_tokens_per_frame,
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

                    backbone_lr = model_scheduler.get_last_lr()[0]
                    proj_lr     = proj_scheduler.get_last_lr()[0]

                    log.info(
                        "epoch %d  step %d  loss %.4f"
                        "  backbone_lr %.2e  proj_lr %.2e",
                        epoch, global_step, avg_loss,
                        backbone_lr, proj_lr,
                    )
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    step_metrics.update({
                        "train/loss":        avg_loss,
                        "train/backbone_lr": backbone_lr,
                        "train/proj_lr":     proj_lr,
                    })

                if use_wandb and step_metrics and is_main:
                    wandb.log(step_metrics, step=global_step)

        # ── Validation (loss) ─────────────────────────────────────────────────
        model.eval()
        projector.eval()
        val_loss_sum = torch.zeros(1, device=device)
        n_batches    = torch.zeros(1, device=device)

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation", leave=False,
                           disable=not is_main, file=sys.stdout)
            val_bar.refresh()
            for batch in val_bar:
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

        # ── Generative validation (full val set, end of epoch) ────────────────
        if is_main:
            log.info("Running end-of-epoch generative validation …")
        gen_results = _run_generative_validation(
            model, projector, val_ds, processor, tokenizer, accelerator,
            pose_tokens_per_frame=args.pose_tokens_per_frame,
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

        # ── Checkpointing ─────────────────────────────────────────────────────
        ckpt_dir = output_dir / f"epoch-{epoch}"
        if is_main:
            ckpt_dir.mkdir(exist_ok=True)
        accelerator.wait_for_everyone()

        proj_state = accelerator.get_state_dict(projector)
        if is_main:
            torch.save(proj_state, ckpt_dir / "pose_projector.pt")
            _save_config(ckpt_dir, feat_dim, hidden_size, args)

        new_best = val_loss < best_val_loss
        if new_best:
            best_val_loss = val_loss
            best_dir = output_dir / "best"
            if is_main:
                best_dir.mkdir(exist_ok=True)
                torch.save(proj_state, best_dir / "pose_projector.pt")
                _save_config(best_dir, feat_dim, hidden_size, args)
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
            best_val_loss=best_val_loss, feat_dim=feat_dim, hidden_size=hidden_size,
            tokens_per_frame=args.pose_tokens_per_frame,
        )

    if is_main:
        log.info("Training complete. Best val_loss: %.4f", best_val_loss)
        if use_wandb:
            wandb.finish()


def _save_config(directory: Path, feat_dim: int, hidden_size: int, args) -> None:
    cfg = {
        "pose_feature_dim":      feat_dim,
        "hidden_size":           hidden_size,
        "pose_tokens_per_frame": args.pose_tokens_per_frame,
        "model_name":            args.model_name,
        "backbone_lr":           args.backbone_lr,
        "lr":                    args.lr,
        "pose_loss_weight":      args.pose_loss_weight,
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
    """Save accelerator state (model + projector + optimizer + scheduler + RNG)
    plus a small train_state.json for bookkeeping."""
    ckpt_dir = Path(ckpt_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving resume checkpoint to %s …", ckpt_dir)
    _reset_fsdp_training_state(accelerator)
    accelerator.save_state(str(ckpt_dir / "accel_state"))
    if accelerator.is_main_process:
        state = {
            "next_epoch":             next_epoch,
            "global_step":            global_step,
            "step_in_epoch":          step_in_epoch,
            "best_val_loss":          best_val_loss,
            "feat_dim":               feat_dim,
            "hidden_size":            hidden_size,
            "pose_tokens_per_frame":  tokens_per_frame,
        }
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(state, f, indent=2)
        log.info("Resume checkpoint saved (next_epoch=%d, global_step=%d)",
                 next_epoch, global_step)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Qwen3.5-VL on QVID+Kinetics+HMDB51+ShareGPT4Video with pose features"
    )
    # Model
    p.add_argument("--model_name",   default="Qwen/Qwen3.5-9B")
    p.add_argument("--output_dir",   default="/orcd/compute/ppliang/001/qwen_multi_big")
    p.add_argument("--resume_ckpt", default="/orcd/compute/ppliang/001/qwen_multi_big/resume_ckpt")
    # p.add_argument("--resume_ckpt",  default="/home/ixzhu/orcd/scratch/qwen_pose/run3/resume_ckpt",
    #                help="Directory to save/load full resume checkpoints (model + optimizer + scheduler).")

    # Data
    p.add_argument("--pose_feature_dim", type=int, default=None,
                   help="Defaults to MediaPipe TOTAL_DIM (1659) if not set.")
    p.add_argument("--pose_tokens_per_frame", type=int, default=16,
                   help="Number of pose tokens emitted per sampled video frame "
                        "(replaces the old --pose_sample_n).")
    p.add_argument("--fps",             type=float, default=1.0,
                   help="Target sampling rate (frames/sec) for video frames fed to the vision encoder.")
    p.add_argument("--max_frames",      type=int,   default=16,
                   help="Maximum number of video frames per clip (caps fps-based sampling).")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=4,
                   help="DataLoader worker processes for parallel video decoding.")

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
    p.add_argument("--torch_compile",  action="store_true", default=False,
                   help="Use torch.compile for faster execution. Off by default under FSDP.")
    p.add_argument("--no_torch_compile", dest="torch_compile", action="store_false")
    p.add_argument("--save_full_model", action="store_true", default=True,
                   help="Save full Qwen model at best checkpoint (~14 GB).")
    p.add_argument("--save_steps", type=int, default=10,
                   help="Save resume checkpoint every N training steps (default: 10).")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Run generative validation every N training steps (0 = only at end of epoch).")
    p.add_argument("--val_max_samples", type=int, default=200,
                   help="Max samples for mid-training generative validation (0 = all). "
                        "End-of-epoch validation always uses the full set.")
    p.add_argument("--val_max_new_tokens", type=int, default=32,
                   help="Max tokens to generate per sample during generative validation.")

    # Logging
    p.add_argument("--log_steps",       type=int,  default=10)
    p.add_argument("--wandb_project",   type=str,  default="qwen-multi-video")
    p.add_argument("--wandb_run_name",  type=str,  default=None)
    p.add_argument("--no_wandb",        action="store_true",
                   help="Disable WandB logging.")
    p.add_argument("--log_dir",         type=str, default="/orcd/compute/ppliang/001/qwen_multi_new/logs",
                   help="Directory for run log files.  A timestamped "
                        "run_YYYYMMDD_HHMMSS.log is created automatically.  "
                        "Defaults to <output_dir>/logs.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
