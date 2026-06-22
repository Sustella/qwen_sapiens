#!/usr/bin/env python3
"""
train_avasd_no_pose.py — Fine-tune a Qwen3.5-VL checkpoint on the autism
dataset mix (av_asd + scrape_asd) WITHOUT pose features.

Pairs train_baseline.py with train_avasd.py: same finetuning recipe as
train_avasd.py (autism data, 4-metric av_asd scoring, end-of-epoch
generative validation) but with NO pose projector and NO pose-token
injection — analogous to how train_baseline.py strips the projector
out of train.py.

Typical use: start from a backbone already fine-tuned by train_baseline.py
(default --model_name points at qwen_multi_base/best/model) and adapt it
to the autism dataset mix with the same hyperparameters as train_avasd.py.

Usage
-----
  accelerate launch train_avasd_no_pose.py \\
      --model_name /orcd/compute/ppliang/001/qwen_multi_base/resume_ckpt/model \\
      --output_dir /orcd/compute/ppliang/001/qwen_autism_no_pose
"""

import argparse
import functools
import json
import logging
import math
import os

# Reduce CUDA allocator fragmentation. Variable seq_len per iter (visual
# tokens swing 10x across samples) strands big reserved-but-unallocated
# chunks under the default allocator; expandable_segments lets blocks grow.
# Must be set before torch.cuda touches anything.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
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

from multi_dataset import get_av_asd_samples, get_scrape_asd_samples


def get_autism_samples():
    """Combine av_asd + scrape_asd into the train/val/eval split shape used by
    the rest of the pipeline (matches multi_dataset.get_all_samples)."""
    av = get_av_asd_samples()
    sa = get_scrape_asd_samples()
    return {
        split: av.get(split, []) + sa.get(split, [])
        for split in ("train", "val", "eval")
    }


def _is_video_readable(
    video_path: str,
    max_visual_tokens: Optional[int] = None,
    max_frames_for_budget: int = 16,
) -> bool:
    """Quick check: cv2 can open the file and it has at least one frame.
    Same visual-token guard as train_avasd.py — drop high-res videos whose
    lm_head activations would dominate per-sample memory under FSDP."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
            return False
        if max_visual_tokens is not None and max_visual_tokens > 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0:
                import math as _m
                tokens_per_frame = _m.ceil(w / 28) * _m.ceil(h / 28)
                est_tokens = tokens_per_frame * max_frames_for_budget
                if est_tokens > max_visual_tokens:
                    return False
        return True
    finally:
        cap.release()


# ── Answer matching / AVASD multi-label scoring ──────────────────────────────

def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_mc_letter(text: str) -> str:
    m = re.search(r"\b([A-Ea-e])\b", text)
    if m:
        return m.group(1).upper()
    c = text.strip()[:1].upper()
    return c if c in "ABCDE" else ""


def _multilabel_set(text: str) -> set:
    """Split a comma-separated multi-label answer into a set of normalized
    option strings. Order- and whitespace-insensitive."""
    return {_normalize(p) for p in text.split(",") if _normalize(p)}


# The 9 canonical av_asd behaviors, normalized via _normalize().
_AVASD_BEHAVIORS_RAW = (
    "Absence or Avoidance of Eye Contact",
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Non-Responsiveness to Verbal Interaction",
    "Non-Typical Language",
    "Object Lining-Up",
    "Self-Hitting or Self-Injurious Behavior",
    "Self-Spinning or Spinning Objects",
    "Upper Limb Stereotypies",
)
_AVASD_BEHAVIORS = frozenset(_normalize(b) for b in _AVASD_BEHAVIORS_RAW)

_METRIC_KEYS = ("strict", "f1", "hamming", "balanced_acc")


def _score_one(pred: str, gt: str, task_type: str) -> dict:
    """Per-sample scores: strict / f1 / hamming / balanced_acc. Identical to
    train_avasd._score_one — kept here so the no-pose run uses the same val
    metrics for direct comparison with the pose-mode run."""
    if task_type == "multiple_choice":
        ok = float(_extract_mc_letter(pred) == gt.strip().upper())
        return {"strict": ok, "f1": ok, "hamming": ok, "balanced_acc": ok}

    pred_set = _multilabel_set(pred)
    gt_set   = _multilabel_set(gt)

    strict = float(pred_set == gt_set)

    if not pred_set and not gt_set:
        f1 = 1.0
    elif not pred_set or not gt_set:
        f1 = 0.0
    else:
        inter = len(pred_set & gt_set)
        if inter == 0:
            f1 = 0.0
        else:
            p = inter / len(pred_set)
            r = inter / len(gt_set)
            f1 = 2 * p * r / (p + r)

    universe = _AVASD_BEHAVIORS
    pp = pred_set & universe
    gg = gt_set   & universe
    tp = len(pp & gg)
    fp = len(pp - gg)
    fn = len(gg - pp)
    tn = len(universe) - tp - fp - fn

    hamming = (tp + tn) / len(universe)

    sens = tp / (tp + fn) if (tp + fn) > 0 else None
    spec = tn / (tn + fp) if (tn + fp) > 0 else None
    if sens is not None and spec is not None:
        balanced_acc = (sens + spec) / 2
    elif sens is not None:
        balanced_acc = sens
    elif spec is not None:
        balanced_acc = spec
    else:
        balanced_acc = 1.0

    return {"strict": strict, "f1": f1, "hamming": hamming, "balanced_acc": balanced_acc}


def _zero_metric_acc() -> dict:
    return {m: 0.0 for m in _METRIC_KEYS} | {"total": 0}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Dataset (video frames only, no pose extraction) ──────────────────────────

class VideoOnlyDataset(Dataset):
    """Video-only counterpart to train_avasd.MultiVideoDataset: samples frames
    at target fps but skips MediaPipe pose extraction entirely. Drops
    unreadable / oversized videos at init via a one-time pre-scan."""

    def __init__(
        self,
        samples: List[dict],
        fps: float = 2.0,
        max_frames: int = 32,
        skip_unreadable: bool = True,
        scan_desc: str = "Scanning videos",
        max_visual_tokens: Optional[int] = None,
    ):
        self.fps = fps
        self.max_frames = max_frames

        if skip_unreadable:
            self.examples: List[dict] = []
            n_drop = 0
            for s in tqdm(samples, desc=scan_desc, file=sys.stdout):
                if _is_video_readable(
                    s["video_path"],
                    max_visual_tokens=max_visual_tokens,
                    max_frames_for_budget=max_frames,
                ):
                    self.examples.append(s)
                else:
                    n_drop += 1
            if n_drop:
                log.warning(
                    "VideoOnlyDataset: dropped %d / %d unreadable / oversized videos",
                    n_drop, len(samples),
                )
        else:
            self.examples = list(samples)

        log.info("VideoOnlyDataset: %d examples loaded", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        s = self.examples[idx]
        pil_images = _sample_frames(s["video_path"], self.fps, self.max_frames)
        return {
            "pil_images": pil_images,
            "question":   s["question"],
            "answer":     s["answer"],
            "dataset":    s["dataset"],
            "task_type":  s["task_type"],
        }


def _sample_frames(video_path: str, fps: float, max_frames: int) -> List[Image.Image]:
    """Sample frames from a video at `fps` frames-per-second, capped at `max_frames`."""
    import cv2

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


# ── Collate (no pose tokens) ────────────────────────────────────────────────

def _collate(batch: List[dict], processor, tokenizer) -> dict:
    """Build processor inputs for a batch. Labels are -100 for prompt
    positions; only answer tokens have real labels."""
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

    answer_lens = []
    for full_t, prompt_t in zip(full_texts, prompt_texts):
        answer_text = full_t[len(prompt_t):]
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        answer_lens.append(len(answer_ids))

    inputs = processor(
        text=full_texts, images=all_images, return_tensors="pt", padding=True,
    )
    input_ids = inputs["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, seq_len = input_ids.shape
    labels = torch.full((B, seq_len), -100, dtype=input_ids.dtype)

    for i, ans_len in enumerate(answer_lens):
        nonpad = (input_ids[i] != pad_id).sum().item()
        ans_start = max(1, nonpad - ans_len)
        labels[i, ans_start:nonpad] = input_ids[i, ans_start:nonpad]

    result = {
        "input_ids":     input_ids,
        "attention_mask": inputs["attention_mask"],
        "labels":        labels,
    }
    if "mm_token_type_ids" in inputs:
        result["mm_token_type_ids"] = inputs["mm_token_type_ids"]
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in inputs:
            result[k] = inputs[k]
    return result


def _collate_gen(batch: List[dict], processor, tokenizer) -> dict:
    """Build prompt-only inputs for generative validation (no pose tokens)."""
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
        "questions":  [ex["question"]  for ex in batch],
    }


# ── Generative validation ────────────────────────────────────────────────────

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
    max_new_tokens: int = 64, max_samples: int = 0,
    print_examples: int = 0,
) -> dict:
    """Generate (plain model.generate, no projector) and compute the four
    AVASD metrics — same scoring as train_avasd._run_generative_validation."""
    is_main = accelerator.is_main_process

    subset = _stratified_subset(val_ds, max_samples)

    collate_fn = lambda b: _collate_gen(b, processor, tokenizer)
    gen_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)
    gen_loader = accelerator.prepare(gen_loader)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    overall    = _zero_metric_acc()
    by_dataset = defaultdict(_zero_metric_acc)
    by_tasktype = defaultdict(_zero_metric_acc)

    was_training = model.training
    model.eval()

    _world = accelerator.num_processes
    _global_total = len(gen_loader) * _world
    with torch.no_grad():
        gen_bar = tqdm(
            desc="Gen-val", leave=False,
            disable=not is_main,
            total=_global_total,
            file=sys.stdout,
        )
        gen_bar.refresh()
        if is_main:
            print(f"[gen-val] starting, global_total={_global_total} "
                  f"({len(gen_loader)} batches/rank × {_world} ranks)",
                  flush=True)
        examples_remaining = print_examples
        for _gv_idx, batch in enumerate(gen_loader):
            if is_main:
                gen_bar.update(_world)
                if gen_bar.n > gen_bar.total:
                    gen_bar.n = gen_bar.total
            answers    = batch.pop("answers")
            datasets   = batch.pop("datasets")
            task_types = batch.pop("task_types")
            questions  = batch.pop("questions", [None] * len(answers))

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

            if is_main and examples_remaining > 0:
                for q, pred, gt, ds, tt in zip(questions, preds, answers, datasets, task_types):
                    if examples_remaining <= 0:
                        break
                    log.info("[gen-val example] dataset=%s  task_type=%s", ds, tt)
                    if q:
                        q_show = q if len(q) <= 400 else q[:400] + "…"
                        log.info("  question : %s", q_show.replace("\n", " ⏎ "))
                    log.info("  gt       : %s", gt)
                    log.info("  pred     : %s", pred)
                    examples_remaining -= 1

            for pred, gt, ds, tt in zip(preds, answers, datasets, task_types):
                scores = _score_one(pred, gt, tt)
                for bucket in (overall, by_dataset[ds], by_tasktype[tt]):
                    for m in _METRIC_KEYS:
                        bucket[m] += scores[m]
                    bucket["total"] += 1

    if was_training:
        model.train()

    def acc(s, k):
        return s[k] / max(1, s["total"])

    return {
        "total": overall["total"],
        **{f"overall_{k}": acc(overall, k) for k in _METRIC_KEYS},
        "by_dataset": {
            ds: {**{k: acc(s, k) for k in _METRIC_KEYS}, "total": s["total"]}
            for ds, s in sorted(by_dataset.items())
        },
        "by_task_type": {
            tt: {**{k: acc(s, k) for k in _METRIC_KEYS}, "total": s["total"]}
            for tt, s in sorted(by_tasktype.items())
        },
    }


def _log_gen_results(label: str, results: dict, log_fn=log.info):
    """Pretty-print generative validation results across all four AVASD metrics."""
    log_fn(
        "%s: strict=%.4f  f1=%.4f  hamming=%.4f  bal_acc=%.4f  (n=%d)",
        label,
        results["overall_strict"], results["overall_f1"],
        results["overall_hamming"], results["overall_balanced_acc"],
        results["total"],
    )
    for ds, s in results["by_dataset"].items():
        log_fn(
            "    %-12s strict=%.4f  f1=%.4f  hamming=%.4f  bal_acc=%.4f  (n=%d)",
            ds + ":", s["strict"], s["f1"], s["hamming"], s["balanced_acc"], s["total"],
        )
    for tt, s in results.get("by_task_type", {}).items():
        log_fn(
            "    %-16s strict=%.4f  f1=%.4f  hamming=%.4f  bal_acc=%.4f  (n=%d)",
            tt + ":", s["strict"], s["f1"], s["hamming"], s["balanced_acc"], s["total"],
        )


# ── Model loading (HF-format or accel-checkpoint-format) ─────────────────────

def _load_model_and_processor(model_path: str, base_model: str):
    """Load a Qwen3.5-VL model + processor, accepting either format:

      • HF-format dir or hub name (config.json + model.safetensors at top), or
      • an accelerate checkpoint dir whose weights live under ``accel_state/``
        (non-sharded ``model.safetensors`` from a train_baseline.py-style run,
        or sharded ``pytorch_model_fsdp_0/`` from an FSDP run).

    For the accel-format case the architecture/tokenizer come from *base_model*
    and the trained weights are overlaid. Mirrors
    ``evaluate_multidataset.load_model_from_accel_checkpoint`` — kept local so
    this training script doesn't drag in the eval module's import chain.
    """
    p = Path(model_path)
    accel_dir = p / "accel_state"
    has_hf_weights_top = (
        (p / "model.safetensors").exists()
        or (p / "model.safetensors.index.json").exists()
        or (p / "pytorch_model.bin").exists()
        or (p / "pytorch_model.bin.index.json").exists()
    )
    is_accel = p.is_dir() and accel_dir.is_dir() and not has_hf_weights_top

    if not is_accel:
        log.info("Loading HF-format model: %s", model_path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model, processor

    log.info(
        "Detected accelerate-format checkpoint at %s — loading base architecture "
        "from %s and overlaying accel_state weights", model_path, base_model,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    shard_dir       = accel_dir / "pytorch_model_fsdp_0"
    safetensors_pth = accel_dir / "model.safetensors"
    bin_pth         = accel_dir / "pytorch_model.bin"

    if shard_dir.exists():
        log.info("Overlaying FSDP-sharded weights from %s", shard_dir)
        from evaluate_multidataset import _load_dcp_state_dict
        loaded = _load_dcp_state_dict(model.state_dict(), shard_dir)
        model.load_state_dict(loaded, strict=False)
        model.tie_weights()
        del loaded
    elif safetensors_pth.exists():
        log.info("Overlaying non-sharded weights from %s", safetensors_pth)
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_pth))
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        del state_dict
    elif bin_pth.exists():
        log.info("Overlaying non-sharded weights from %s", bin_pth)
        state_dict = torch.load(str(bin_pth), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        del state_dict
    else:
        raise FileNotFoundError(
            f"No weights found in {accel_dir} — expected one of: "
            "pytorch_model_fsdp_0/, model.safetensors, or pytorch_model.bin"
        )

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    return model, processor


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

    # ── WandB (main process only) ────────────────────────────────────────────
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

    # ── Resume checkpoint detection ──────────────────────────────────────────
    resume_ckpt = Path(args.resume_ckpt)
    resume_state = None
    if (resume_ckpt / "train_state.json").exists():
        with open(resume_ckpt / "train_state.json") as f:
            resume_state = json.load(f)
        if is_main:
            log.info("Found resume checkpoint at %s (next_epoch=%d)",
                     resume_ckpt, resume_state["next_epoch"])

    # ── Model ────────────────────────────────────────────────────────────────
    # --model_name may be an HF-format dir/hub name OR an accelerate
    # checkpoint dir (e.g. .../qwen_multi_base/resume_ckpt) whose weights sit
    # under accel_state/. The helper auto-detects and, in the accel case,
    # initializes from --base_model then overlays the accel_state weights.
    # accelerator.load_state() further overlays trained weights when resuming
    # via --resume_ckpt below.
    log.info("Loading model: %s", args.model_name)
    model, processor = _load_model_and_processor(args.model_name, args.base_model)
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

    # ── Datasets ─────────────────────────────────────────────────────────────
    splits = get_autism_samples()
    train_ds = VideoOnlyDataset(
        splits["train"], fps=args.fps, max_frames=args.max_frames,
        scan_desc="Scanning train videos",
        max_visual_tokens=args.max_visual_tokens,
    )
    val_ds = VideoOnlyDataset(
        splits["val"], fps=args.fps, max_frames=args.max_frames,
        scan_desc="Scanning val videos",
        max_visual_tokens=args.max_visual_tokens,
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

    # ── Prepare (FSDP wrap) ──────────────────────────────────────────────────
    if is_main:
        print("[stage] accelerator.prepare(model, loaders) — FSDP wrap …", flush=True)
    model, train_loader, val_loader = accelerator.prepare(
        model, train_loader, val_loader
    )
    if is_main:
        print("[stage] accelerator.prepare(model, …) done.", flush=True)

    if args.torch_compile:
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )

    # accelerate's prepared scheduler advances its internal counter by
    # num_processes per .step() call, so scale total_steps to match — otherwise
    # the cosine exhausts to LR=0 in 1/num_processes of the intended schedule.
    total_steps  = (
        math.ceil(len(train_loader) / args.grad_accum)
        * args.num_epochs
        * accelerator.num_processes
    )
    warmup_steps = round(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    # ── Training loop ────────────────────────────────────────────────────────
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

    # ── Warm up FSDP ─────────────────────────────────────────────────────────
    # FSDP lazy-initializes on the first forward through the ROOT wrapper.
    # model.generate() bypasses the root, so trigger init through model() first.
    if is_main:
        log.info("Initializing FSDP …")
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        model(input_ids=_dummy, attention_mask=torch.ones_like(_dummy))
    accelerator.wait_for_everyone()

    # ── Pre-training baseline (fresh runs only, opt-in) ─────────────────────
    if not resume_state and not args.skip_baseline_eval:
        if is_main:
            log.info("Fresh run — evaluating pretrained baseline on validation …")
        pt_results = _run_generative_validation(
            model, val_ds, processor, tokenizer, accelerator,
            max_new_tokens=args.val_max_new_tokens,
            max_samples=args.val_max_samples,
            print_examples=1,
        )
        if is_main:
            _log_gen_results("Baseline (pretrained, no pose)", pt_results)
            if use_wandb:
                wandb.log({
                    f"baseline/pretrained_{m}": pt_results[f"overall_{m}"]
                    for m in _METRIC_KEYS
                }, step=0)

    # ── Training epochs ──────────────────────────────────────────────────────
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

        _world = accelerator.num_processes
        pbar = tqdm(
            desc=f"Epoch {epoch}/{args.num_epochs}",
            leave=True, disable=not is_main,
            initial=step_offset * _world,
            total=len(train_loader) * _world,
            file=sys.stdout,
        )
        pbar.refresh()

        for local_step, batch in enumerate(active_loader):
            step = local_step + step_offset
            if is_main:
                pbar.update(_world)
                if pbar.n > pbar.total:
                    pbar.n = pbar.total
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
                            f"val_gen/accuracy_{m}": gen_results[f"overall_{m}"]
                            for m in _METRIC_KEYS
                        })
                        for ds, s in gen_results["by_dataset"].items():
                            for m in _METRIC_KEYS:
                                step_metrics[f"val_gen/{ds}_{m}"] = s[m]

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

        # ── Validation (loss) ────────────────────────────────────────────────
        model.eval()
        val_loss_sum = torch.zeros(1, device=device)
        n_batches    = torch.zeros(1, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False,
                              disable=not is_main, file=sys.stdout):
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

        # ── Generative validation (full val set, end of epoch) ───────────────
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
                _ep = {f"val_gen_epoch/accuracy_{m}": gen_results[f"overall_{m}"]
                       for m in _METRIC_KEYS}
                for ds, s in gen_results["by_dataset"].items():
                    for m in _METRIC_KEYS:
                        _ep[f"val_gen_epoch/{ds}_{m}"] = s[m]
                _ep["epoch"] = epoch
                wandb.log(_ep, step=global_step)

        # ── Checkpointing ────────────────────────────────────────────────────
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
        description="Fine-tune Qwen3.5-VL on the autism mix (av_asd + scrape_asd) "
                    "WITHOUT pose features. Default --model_name points at the "
                    "train_baseline.py output so this picks up where the no-pose "
                    "multi-dataset finetune left off."
    )
    # Model
    p.add_argument(
        "--model_name",
        default="/orcd/compute/ppliang/001/qwen_multi_base/resume_ckpt",
        help="Backbone init: either an HF model dir / hub name (config.json + "
             "model.safetensors at top), or an accelerate-checkpoint dir whose "
             "weights live under accel_state/ (e.g. .../qwen_multi_base/resume_ckpt). "
             "In the accel case --base_model supplies the architecture/tokenizer.",
    )
    p.add_argument(
        "--base_model", default="Qwen/Qwen3.5-9B",
        help="HF model dir / hub name used as the architecture+tokenizer source "
             "when --model_name is an accelerate-format checkpoint (no top-level "
             "model.safetensors / config.json). Ignored for HF-format --model_name.",
    )
    p.add_argument("--output_dir",  default="/orcd/compute/ppliang/001/qwen_autism_no_pose")
    p.add_argument("--resume_ckpt", default="/orcd/compute/ppliang/001/qwen_autism_no_pose/resume_ckpt")

    # Data
    p.add_argument("--fps",             type=float, default=1.0,
                   help="Target sampling rate (frames/sec) for video frames fed to the vision encoder.")
    p.add_argument("--max_frames",      type=int,   default=16,
                   help="Maximum number of video frames per clip (caps fps-based sampling).")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=4,
                   help="DataLoader worker processes for parallel video decoding.")
    p.add_argument("--max_visual_tokens", type=int, default=40000,
                   help="Drop any video whose estimated Qwen-VL visual-token "
                        "count ceil(W/28)*ceil(H/28)*max_frames exceeds this. "
                        "Same guard as train_avasd.py.")

    # Training
    p.add_argument("--num_epochs",   type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--grad_accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-5,
                   help="Learning rate (matches train_avasd.py backbone_lr).")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--torch_compile",  action="store_true", default=False)
    p.add_argument("--no_torch_compile", dest="torch_compile", action="store_false")
    p.add_argument("--save_full_model", action="store_true", default=True,
                   help="Save full Qwen model at best checkpoint (~14 GB).")
    p.add_argument("--skip_baseline_eval", action="store_true", default=True,
                   help="Skip the pre-training baseline gen-val pass. Default: True.")
    p.add_argument("--no_skip_baseline_eval", dest="skip_baseline_eval",
                   action="store_false")
    p.add_argument("--save_steps", type=int, default=1,
                   help="Save resume checkpoint every N global optimizer steps. "
                        "Default 1 so an OOM mid-epoch doesn't lose progress.")
    p.add_argument("--val_steps", type=int, default=5,
                   help="Run generative validation every N steps (0 = only at end of epoch).")
    p.add_argument("--val_max_samples", type=int, default=200,
                   help="Max samples for mid-training gen-val. End-of-epoch always uses the full set.")
    p.add_argument("--val_max_new_tokens", type=int, default=64,
                   help="Max tokens to generate per sample during gen-val. "
                        "av_asd multi-label answers can exceed 32 tokens.")

    # Logging
    p.add_argument("--log_steps",       type=int,  default=1)
    p.add_argument("--wandb_project",   type=str,  default="qwen-avasd-no-pose")
    p.add_argument("--wandb_run_name",  type=str,  default=None)
    p.add_argument("--no_wandb",        action="store_true")
    p.add_argument("--log_dir",         type=str, default=None,
                   help="Directory for run log files. Defaults to <output_dir>/logs.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
