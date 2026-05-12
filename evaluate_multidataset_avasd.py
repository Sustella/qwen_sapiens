#!/usr/bin/env python3
"""
evaluate_multidataset_avasd.py — Evaluate Qwen checkpoints on the AV-ASD test
split with full auto-regressive generation.

Supports three modes (any combination):

  pretrained      : plain Qwen3.5-VL (no pose projector, no fine-tune).  Sees
                    video frames + prompt; the model answers "0"/"1" directly.
  pose_projector  : the pose-augmented checkpoint from train.py.  Uses the
                    trained pose projector to inject pose tokens at the
                    prompt/answer boundary (same layout as training).
  finetuned       : the AV-ASD-fine-tuned checkpoint from train_avasd.py.
                    Can be evaluated with the (frozen) projector that was used
                    during fine-tuning, OR plain (video-only), controlled by
                    --finetuned_use_projector.

Metrics mirror evaluate_avasd.py's ``compute_metrics`` (accuracy, balanced
accuracy, F1, precision, recall, TPR/TNR, confusion matrix, unparsable count),
reported overall and per-behavior.

Usage
-----
  # All three, side-by-side
  python evaluate_multidataset_avasd.py \\
      --modes pretrained pose_projector finetuned \\
      --pose_checkpoint     /orcd/compute/ppliang/001/qwen_multi/best \\
      --finetuned_checkpoint /orcd/compute/ppliang/001/qwen_avasd/best \\
      --pretrained_model Qwen/Qwen3.5-9B

  # Pose-projector only
  python evaluate_multidataset_avasd.py \\
      --modes pose_projector \\
      --pose_checkpoint /orcd/compute/ppliang/001/qwen_multi/best

  # Fine-tuned without projector (video + prompt only)
  python evaluate_multidataset_avasd.py \\
      --modes finetuned \\
      --finetuned_checkpoint /orcd/compute/ppliang/001/qwen_avasd/best \\
      --no_finetuned_use_projector
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from train import MultiVideoDataset, PoseFeatureProjector, _make_pose_hook
from evaluate_multidataset import (
    load_model,
    load_model_from_accel_checkpoint,
    load_projector_from_hf,
    load_projector_from_accel,
    _detect_checkpoint_format,
    _collate_gen as _collate_gen_with_pose,
    _collate_gen_vanilla,
)
from avasd_dataset import get_avasd_samples, BEHAVIORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class AVASDEvalDataset(MultiVideoDataset):
    """MultiVideoDataset that also surfaces ``behavior`` + ``video_id``."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        ex = self.examples[idx]
        item["behavior"] = ex["behavior"]
        item["video_id"] = ex["video_id"]
        return item


# ── Collates that pass through behavior + video_id ───────────────────────────

def _collate_gen_avasd_with_pose(batch, processor, embed_device, pose_sample_n):
    out = _collate_gen_with_pose(batch, processor, embed_device, pose_sample_n)
    out["behaviors"] = [ex["behavior"] for ex in batch]
    out["video_ids"] = [ex["video_id"] for ex in batch]
    return out


def _collate_gen_avasd_vanilla(batch, processor, embed_device):
    out = _collate_gen_vanilla(batch, processor, embed_device)
    out["behaviors"] = [ex["behavior"] for ex in batch]
    out["video_ids"] = [ex["video_id"] for ex in batch]
    return out


# ── Binary metrics (matches evaluate_avasd.py) ───────────────────────────────

def _parse_binary(pred: str) -> Optional[int]:
    pred = (pred or "").strip()
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


def _compute_metrics(y_true: List[int], y_pred: List[Optional[int]]) -> dict:
    tp = fp = fn = tn = 0
    unparsable = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            unparsable += 1
            if t == 1:
                fn += 1
            else:
                fp += 1
            continue
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        else: tn += 1
    total = tp + fp + fn + tn
    acc  = (tp + tn) / total if total > 0 else 0.0
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total, "unparsable": unparsable,
        "accuracy": acc, "balanced_accuracy": (tpr + tnr) / 2,
        "precision": prec, "recall": tpr, "f1": f1,
        "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr,
    }


# ── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_with_projector(model, projector, batch, max_new_tokens, pad_id, eos_id):
    pose_feat = batch.pop("pose_feat").to(dtype=torch.bfloat16)
    pose_positions = batch.pop("pose_positions")

    pose_embeds = projector(pose_feat.to(next(projector.parameters()).device))

    lang_model = model.model.language_model
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
    finally:
        handle.remove()
    return new_ids


@torch.no_grad()
def evaluate_avasd(model, processor, loader, max_new_tokens: int,
                   projector=None) -> dict:
    """Generate on the AV-ASD loader and compute overall + per-behavior metrics."""
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    model.eval()
    if projector is not None:
        projector.eval()

    by_behavior = defaultdict(lambda: {"y_true": [], "y_pred": [], "samples": []})
    all_true: List[int] = []
    all_pred: List[Optional[int]] = []
    all_samples = []

    pbar = tqdm(loader, desc="Generating", dynamic_ncols=True)
    for batch in pbar:
        answers   = batch.pop("answers")
        # Strip the multi_dataset fields (present from the shared collate)
        batch.pop("datasets", None)
        batch.pop("task_types", None)
        behaviors = batch.pop("behaviors")
        video_ids = batch.pop("video_ids")

        if projector is not None and "pose_feat" in batch:
            new_ids = _generate_with_projector(
                model, projector, batch, max_new_tokens, pad_id, eos_id,
            )
        else:
            batch.pop("pose_feat", None)
            batch.pop("pose_positions", None)
            prompt_len = batch["input_ids"].shape[1]
            gen_kwargs = {k: v for k, v in batch.items() if v is not None}
            gen_ids = model.generate(
                **gen_kwargs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            new_ids = gen_ids[:, prompt_len:]

        preds_text = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in new_ids
        ]

        for pt_text, gt, behavior, vid in zip(preds_text, answers, behaviors, video_ids):
            parsed = _parse_binary(pt_text)
            t = int(gt)
            by_behavior[behavior]["y_true"].append(t)
            by_behavior[behavior]["y_pred"].append(parsed)
            all_true.append(t)
            all_pred.append(parsed)
            record = {
                "video_id": vid,
                "behavior": behavior,
                "gt":       t,
                "pred_raw": pt_text,
                "pred":     parsed,
            }
            by_behavior[behavior]["samples"].append(record)
            all_samples.append(record)

        overall_stats = _compute_metrics(all_true, all_pred)
        pbar.set_postfix(
            acc=f"{overall_stats['accuracy']:.4f}",
            bal=f"{overall_stats['balanced_accuracy']:.4f}",
        )

    per_behavior = {
        b: _compute_metrics(d["y_true"], d["y_pred"])
        for b, d in sorted(by_behavior.items())
    }
    overall = _compute_metrics(all_true, all_pred)

    return {
        "overall":      overall,
        "by_behavior":  per_behavior,
        "samples":      all_samples,
    }


# ── Pretty printing ──────────────────────────────────────────────────────────

def print_results(label: str, results: dict) -> None:
    W = 100
    print()
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    ov = results["overall"]
    print(f"\n  Samples: {ov['total']}  "
          f"(Pos: {ov['tp'] + ov['fn']}, Neg: {ov['tn'] + ov['fp']}, "
          f"Unparsable: {ov['unparsable']})")
    print(f"\n  Accuracy          : {ov['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {ov['balanced_accuracy']:.4f}")
    print(f"  F1 Score          : {ov['f1']:.4f}")
    print(f"  Precision         : {ov['precision']:.4f}")
    print(f"  Recall (TPR)      : {ov['recall']:.4f}")
    print(f"  TNR               : {ov['tnr']:.4f}")
    print(f"  Confusion         : TP={ov['tp']} FP={ov['fp']} "
          f"FN={ov['fn']} TN={ov['tn']}")

    print("\n  Per-behavior:")
    header = f"  {'Behavior':<45} {'Acc':>7} {'Bal.Acc':>8} {'F1':>7} {'TPR':>7} {'TNR':>7} {'N':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b, m in results["by_behavior"].items():
        print(f"  {b:<45} {m['accuracy']:>7.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['f1']:>7.4f} {m['tpr']:>7.4f} {m['tnr']:>7.4f} {m['total']:>5d}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'OVERALL':<45} {ov['accuracy']:>7.4f} {ov['balanced_accuracy']:>8.4f} "
          f"{ov['f1']:>7.4f} {ov['tpr']:>7.4f} {ov['tnr']:>7.4f} {ov['total']:>5d}")
    print("=" * W)


def print_comparison(results_by_mode: dict) -> None:
    """Side-by-side comparison across modes (overall + per-behavior)."""
    modes = list(results_by_mode.keys())
    if len(modes) < 2:
        return

    W = 80 + 14 * len(modes)
    print()
    print("=" * W)
    print("  COMPARISON: " + "  vs  ".join(modes))
    print("=" * W)

    def fmt_row(label, values):
        cells = "  ".join(f"{v:>10}" for v in values)
        print(f"  {label:<45}  {cells}")

    fmt_row("Metric",      [m for m in modes])
    fmt_row("Accuracy",    [f"{results_by_mode[m]['overall']['accuracy']:.4f}"          for m in modes])
    fmt_row("Bal.Accuracy",[f"{results_by_mode[m]['overall']['balanced_accuracy']:.4f}" for m in modes])
    fmt_row("F1",          [f"{results_by_mode[m]['overall']['f1']:.4f}"                for m in modes])
    fmt_row("Precision",   [f"{results_by_mode[m]['overall']['precision']:.4f}"         for m in modes])
    fmt_row("Recall",      [f"{results_by_mode[m]['overall']['recall']:.4f}"            for m in modes])
    fmt_row("Unparsable",  [str(results_by_mode[m]['overall']['unparsable'])            for m in modes])

    print("  " + "-" * (W - 2))
    fmt_row("Per-behavior (balanced accuracy)", [""] * len(modes))
    all_behaviors = sorted({
        b for m in modes for b in results_by_mode[m]["by_behavior"]
    })
    for b in all_behaviors:
        fmt_row(
            b,
            [f"{results_by_mode[m]['by_behavior'].get(b, {}).get('balanced_accuracy', 0.0):.4f}"
             for m in modes],
        )
    print("=" * W)


# ── Model-loading helpers ────────────────────────────────────────────────────

def _load_model_any(ckpt_or_name: str, base_model: Optional[str]):
    """Load a Qwen model from either a plain HF name/dir, an HF-format
    checkpoint (with model/ subdir), or an accelerate checkpoint (with
    accel_state/).  Returns (model, processor, embed_device)."""
    ckpt = Path(ckpt_or_name)
    if ckpt.is_dir() and (ckpt / "model").is_dir():
        # HF-format train checkpoint: load model/ directly
        return load_model(str(ckpt / "model"))
    if ckpt.is_dir() and (ckpt / "accel_state").is_dir():
        if not base_model:
            raise RuntimeError(
                f"{ckpt_or_name} is an accelerate checkpoint; --base_model is required."
            )
        return load_model_from_accel_checkpoint(str(ckpt), base_model)
    # Treat as HF model name / direct path
    return load_model(ckpt_or_name)


def _load_projector_any(ckpt: str, device) -> PoseFeatureProjector:
    fmt = _detect_checkpoint_format(ckpt)
    if fmt == "hf":
        projector = load_projector_from_hf(ckpt, device)
    else:
        projector = load_projector_from_accel(ckpt, device)
    if projector is None:
        raise RuntimeError(f"Could not load projector from {ckpt}")
    return projector


# ── Mode drivers ─────────────────────────────────────────────────────────────

def _build_loader(eval_samples, processor, embed_device, args, with_pose: bool):
    ds = AVASDEvalDataset(
        eval_samples, fps=args.fps, max_frames=args.max_frames,
        pose_sample_n=args.pose_sample_n,
        skip_missing_pose=with_pose,  # only need pose files when actually used
    )
    if with_pose:
        collate_fn = lambda b: _collate_gen_avasd_with_pose(
            b, processor, embed_device, args.pose_sample_n,
        )
    else:
        collate_fn = lambda b: _collate_gen_avasd_vanilla(b, processor, embed_device)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    return ds, loader


def run_mode_pretrained(args, eval_samples) -> dict:
    log.info("─── Mode: pretrained ───")
    model, processor, embed_device = _load_model_any(args.pretrained_model, None)
    ds, loader = _build_loader(eval_samples, processor, embed_device, args,
                               with_pose=False)
    log.info("Evaluating PRETRAINED on %d samples …", len(ds))
    results = evaluate_avasd(model, processor, loader, args.max_new_tokens,
                             projector=None)
    print_results("PRETRAINED MODEL (no projector, no fine-tune)", results)
    del model
    torch.cuda.empty_cache()
    return results


def run_mode_pose_projector(args, eval_samples) -> dict:
    log.info("─── Mode: pose_projector ───")
    model, processor, embed_device = _load_model_any(
        args.pose_checkpoint, args.base_model,
    )
    projector = _load_projector_any(args.pose_checkpoint, embed_device)
    ds, loader = _build_loader(eval_samples, processor, embed_device, args,
                               with_pose=True)
    log.info("Evaluating POSE-PROJECTOR model on %d samples …", len(ds))
    results = evaluate_avasd(model, processor, loader, args.max_new_tokens,
                             projector=projector)
    print_results("POSE-PROJECTOR CUSTOM MODEL (from train.py)", results)
    del model, projector
    torch.cuda.empty_cache()
    return results


def run_mode_finetuned(args, eval_samples) -> dict:
    log.info("─── Mode: finetuned ───")
    model, processor, embed_device = _load_model_any(
        args.finetuned_checkpoint, args.base_model,
    )
    projector = None
    if args.finetuned_use_projector:
        # Prefer the frozen projector saved alongside the fine-tuned checkpoint;
        # fall back to the original pose-projector checkpoint.
        try:
            projector = _load_projector_any(args.finetuned_checkpoint, embed_device)
        except RuntimeError:
            if args.pose_checkpoint:
                log.info("Fine-tuned dir lacks projector — loading from %s",
                         args.pose_checkpoint)
                projector = _load_projector_any(args.pose_checkpoint, embed_device)
            else:
                log.warning(
                    "No projector found alongside fine-tuned checkpoint and "
                    "--pose_checkpoint not given; proceeding without pose tokens."
                )

    ds, loader = _build_loader(
        eval_samples, processor, embed_device, args,
        with_pose=projector is not None,
    )
    label = (
        "FINE-TUNED CUSTOM MODEL (with frozen pose projector)"
        if projector is not None
        else "FINE-TUNED CUSTOM MODEL (no pose — video only)"
    )
    log.info("Evaluating %s on %d samples …", label, len(ds))
    results = evaluate_avasd(model, processor, loader, args.max_new_tokens,
                             projector=projector)
    print_results(label, results)
    del model
    if projector is not None:
        del projector
    torch.cuda.empty_cache()
    return results


MODE_DISPATCH = {
    "pretrained":      run_mode_pretrained,
    "pose_projector":  run_mode_pose_projector,
    "finetuned":       run_mode_finetuned,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Validate args per requested mode
    for mode in args.modes:
        if mode not in MODE_DISPATCH:
            raise SystemExit(f"Unknown mode: {mode}")
    if "pose_projector" in args.modes and not args.pose_checkpoint:
        raise SystemExit("--pose_checkpoint is required for mode=pose_projector")
    if "finetuned" in args.modes and not args.finetuned_checkpoint:
        raise SystemExit("--finetuned_checkpoint is required for mode=finetuned")
    if "pretrained" in args.modes and not args.pretrained_model:
        raise SystemExit("--pretrained_model is required for mode=pretrained")

    log.info("Loading AV-ASD eval split …")
    splits = get_avasd_samples(
        splits=["eval"], anonymized=args.anonymized, require_video=True,
    )
    eval_samples = splits["eval"]
    if args.max_samples is not None:
        eval_samples = eval_samples[:args.max_samples]
        log.info("Limiting eval to %d samples (--max_samples)", args.max_samples)

    output = {"timestamp": timestamp, "args": vars(args)}
    results_by_mode = {}

    for mode in args.modes:
        log.info("Running mode: %s", mode)
        results = MODE_DISPATCH[mode](args, eval_samples)
        results_by_mode[mode] = results
        output[mode] = {
            k: v for k, v in results.items() if k != "samples"
        }
        output[f"{mode}_samples"] = results["samples"]

    if len(args.modes) > 1:
        print_comparison(results_by_mode)

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes_tag = "-".join(args.modes)
    out_path = output_dir / f"eval_avasd_{modes_tag}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", out_path)
    print(f"\nResults saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen checkpoints on the AV-ASD test split."
    )
    p.add_argument(
        "--modes", nargs="+", default=["pose_projector"],
        choices=list(MODE_DISPATCH.keys()),
        help="Which model(s) to evaluate.  Pass one or more of: "
             "pretrained, pose_projector, finetuned.",
    )
    p.add_argument(
        "--pretrained_model", default="Qwen/Qwen3.5-9B",
        help="HF model name or local path for the plain pretrained baseline.",
    )
    p.add_argument(
        "--pose_checkpoint",
        default="/orcd/compute/ppliang/001/qwen_multi/best",
        help="Checkpoint saved by train.py (either HF-format with model/ or "
             "accelerate-format with accel_state/).",
    )
    p.add_argument(
        "--finetuned_checkpoint",
        default="/orcd/compute/ppliang/001/qwen_avasd/best",
        help="Checkpoint saved by train_avasd.py.",
    )
    p.add_argument(
        "--finetuned_use_projector", action="store_true", default=False,
        help="Also inject pose tokens via the frozen projector when evaluating "
             "the fine-tuned model.  Off by default — the 'finetuned' mode is "
             "designed to measure the fine-tuned backbone's performance on "
             "video + prompt alone (the '(no pose)' comparison).",
    )
    p.add_argument(
        "--no_finetuned_use_projector", dest="finetuned_use_projector",
        action="store_false",
    )
    p.add_argument(
        "--base_model", default="Qwen/Qwen3.5-9B",
        help="Base model for accelerate-format checkpoints.",
    )

    # Evaluation knobs
    p.add_argument("--anonymized",    action="store_true", default=False,
                   help="Use the anonymized mesh-overlay clips + matching prompt.")
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--batch_size",     type=int, default=1)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--fps",            type=float, default=2.0)
    p.add_argument("--max_frames",     type=int, default=32)
    p.add_argument("--pose_sample_n",  type=int, default=16)
    p.add_argument(
        "--output_dir",
        default="/orcd/compute/ppliang/001/qwen_avasd/results",
        help="Where to write the timestamped JSON output.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
