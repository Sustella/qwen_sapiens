#!/usr/bin/env python3
"""
evaluate_multidataset.py — Evaluate a Qwen checkpoint on the eval split of
QVID + Kinetics-400 + HMDB51 using full auto-regressive generation.

Modes
-----
  finetuned  : load the fine-tuned backbone from --checkpoint
  pretrained : load the original model from --pretrained_model (no fine-tuning)
  both       : run both and print a side-by-side comparison

Checkpoint formats (auto-detected)
-----------------------------------
  HF format   : --checkpoint points to a dir containing model/ saved via
                save_pretrained (e.g. best/ or epoch-N/ from train.py).
  Accel format: --checkpoint points to a dir containing accel_state/ saved via
                accelerator.save_state (i.e. the resume_ckpt from train.py or
                train_baseline.py).  Supports both FSDP sharded weights
                (pytorch_model_fsdp_0/) and non-sharded weights
                (model.safetensors / pytorch_model.bin).
                Requires --base_model so the architecture can be instantiated
                before loading the checkpoint weights.

Results are printed to stdout and saved to --output_dir/eval_<mode>_<timestamp>.json.

Usage
-----
  # Fine-tuned only — HF checkpoint (has model/ subdir)
  python evaluate_multidataset.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_multi/best

  # Fine-tuned only — accelerate resume checkpoint (has accel_state/)
  python evaluate_multidataset.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_multi/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B

  # Pretrained baseline only
  python evaluate_multidataset.py \\
      --mode pretrained \\
      --pretrained_model Qwen/Qwen3.5-VL-7B-Instruct

  # Side-by-side comparison
  python evaluate_multidataset.py \\
      --mode both \\
      --checkpoint /orcd/compute/ppliang/001/qwen_multi/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B \\
      --pretrained_model Qwen/Qwen3.5-9B
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from train import MultiVideoDataset, PoseFeatureProjector, _make_pose_hook
from multi_dataset import get_all_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Answer normalization ──────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_STOP_WORDS = {"a", "an", "the", "my", "your", "his", "her", "its", "our", "their"}


def normalize_lenient(text: str) -> str:
    """Like normalize() but also strips articles, pronouns, and merges compounds."""
    text = normalize(text)
    words = [w for w in text.split() if w not in _STOP_WORDS]
    text = " ".join(words) if words else text
    # merge compound words: "hair brush" -> "hairbrush"
    text_nospace = text.replace(" ", "")
    return text, text_nospace


def extract_mc_letter(text: str) -> str:
    """
    Extract the first A–E letter from a multiple-choice response.
    Tries word-boundary match first, then falls back to the very first character.
    """
    m = re.search(r"\b([A-Ea-e])\b", text)
    if m:
        return m.group(1).upper()
    c = text.strip()[:1].upper()
    return c if c in "ABCDE" else ""


def is_correct(pred: str, gt: str, task_type: str) -> bool:
    if task_type == "multiple_choice":
        return extract_mc_letter(pred) == gt.strip().upper()
    else:  # open_ended
        return normalize(pred) == normalize(gt)


def is_correct_lenient(pred: str, gt: str, task_type: str) -> bool:
    """
    Lenient matching for open-ended QA.  Counts a prediction as correct if ANY
    of the following hold (after normalization):
      1. Exact match (strict)
      2. After stripping articles/pronouns, exact or compound-merged match
      3. One answer is a substring of the other
      4. The word-sets are identical (handles reordering, e.g. "black and blue"
         vs "blue and black")
    For multiple-choice the behaviour is identical to strict.
    """
    if task_type == "multiple_choice":
        return extract_mc_letter(pred) == gt.strip().upper()

    # 1. Strict
    np, ng = normalize(pred), normalize(gt)
    if np == ng:
        return True

    # 2. Strip stop-words + compound merge
    lp, lp_ns = normalize_lenient(pred)
    lg, lg_ns = normalize_lenient(gt)
    if lp == lg or lp_ns == lg_ns:
        return True

    # 3. Substring containment (either direction)
    if lp in lg or lg in lp:
        return True

    # 4. Word-set equality (handles reordering)
    if set(lp.split()) == set(lg.split()):
        return True

    return False


# ── Collate for generation ────────────────────────────────────────────────────

def _collate_gen(batch: List[dict], processor, embed_device, pose_tokens_per_frame: int = 16) -> dict:
    """
    Build prompt-only inputs with per-frame pose placeholder tokens appended
    at the end (right before where generation would start). Matches the
    per-frame collate used during training so the model sees the same input
    layout.
    """
    texts, all_images = [], []
    for ex in batch:
        content = [{"type": "image"} for _ in ex["pil_images"]]
        content.append({"type": "text", "text": ex["question"]})
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        all_images.append(ex["pil_images"])

    inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"]
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, orig_len = input_ids.shape

    # Pad per-sample pose features to the batch max frame count.
    feats = [ex["pose_feat"] for ex in batch]            # each (N_i, D)
    n_frames_per = [f.shape[0] for f in feats]
    n_max = max(n_frames_per)
    feat_dim = feats[0].shape[-1]
    pose_feat_padded = torch.zeros(B, n_max, feat_dim, dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        pose_feat_padded[i, : f.shape[0]] = f

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
        ap = nonpad  # insert at end of prompt
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
        "input_ids":       new_ids.to(embed_device),
        "attention_mask":   new_mask.to(embed_device),
        "pose_positions":   pose_positions,
        "answers":         [ex["answer"]    for ex in batch],
        "datasets":        [ex["dataset"]   for ex in batch],
        "task_types":      [ex["task_type"] for ex in batch],
        "pose_feat":       pose_feat_padded.to(embed_device),
        "n_frames_per":    n_frames_per,
    }
    if has_mm:
        result["mm_token_type_ids"] = new_mm.to(embed_device)
    for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if k in inputs:
            result[k] = inputs[k].to(embed_device)
    return result


def _collate_gen_vanilla(batch: List[dict], processor, embed_device) -> dict:
    """
    Build processor inputs for the prompt only — no pose placeholder tokens.
    Used for pretrained baseline evaluation.
    """
    texts, all_images = [], []
    for ex in batch:
        content = [{"type": "image"} for _ in ex["pil_images"]]
        content.append({"type": "text", "text": ex["question"]})
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        all_images.append(ex["pil_images"])

    inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )
    result = {
        **{k: v.to(embed_device) for k, v in inputs.items()},
        "answers":    [ex["answer"]    for ex in batch],
        "datasets":   [ex["dataset"]   for ex in batch],
        "task_types": [ex["task_type"] for ex in batch],
    }
    return result


# ── Evaluation loop ───────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {"correct": 0, "total": 0}


@torch.no_grad()
def _generate_with_projector(model, projector, batch, max_new_tokens, pad_id, eos_id):
    """
    Generate using the same hook-based pose injection used during training.

    Pose placeholder tokens are already in the input sequence (inserted by
    _collate_gen).  We project the pose features, register a forward
    pre-hook on the language_model that replaces placeholder embeddings
    with the projected features, then call model.generate().
    """
    pose_feat = batch.pop("pose_feat").to(dtype=torch.bfloat16)
    pose_positions = batch.pop("pose_positions")
    batch.pop("n_frames_per", None)

    pose_embeds = projector(pose_feat.to(next(projector.parameters()).device))

    # Get the language_model sub-module for hook registration
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
def evaluate_model(model, processor, loader, max_new_tokens: int,
                   projector=None) -> dict:
    """
    Generate responses for every batch and compare against ground truth.
    Returns a dict with per-dataset, per-task-type, and overall accuracy,
    plus a full list of per-sample results for offline analysis.

    When *projector* is provided the pose projector predicts the first answer
    token and the backbone continues from there.  Otherwise plain
    model.generate() is used (e.g. for the pretrained baseline).
    """
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    overall          = _empty_stats()
    overall_lenient  = _empty_stats()
    by_dataset       = defaultdict(_empty_stats)
    by_dataset_len   = defaultdict(_empty_stats)
    by_tasktype      = defaultdict(_empty_stats)
    by_tasktype_len  = defaultdict(_empty_stats)
    all_samples = []

    model.eval()
    if projector is not None:
        projector.eval()

    pbar = tqdm(loader, desc="Generating", dynamic_ncols=True)
    for batch in pbar:
        answers    = batch.pop("answers")
        datasets   = batch.pop("datasets")
        task_types = batch.pop("task_types")

        if projector is not None and "pose_feat" in batch:
            new_ids = _generate_with_projector(
                model, projector, batch, max_new_tokens, pad_id, eos_id,
            )
        else:
            # Drop pose-related keys — not needed without projector
            batch.pop("pose_feat", None)
            batch.pop("pose_positions", None)
            batch.pop("n_frames_per", None)
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

        preds = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in new_ids
        ]

        for pred, gt, ds, tt in zip(preds, answers, datasets, task_types):
            correct = is_correct(pred, gt, tt)
            correct_l = is_correct_lenient(pred, gt, tt)
            overall["correct"]             += int(correct)
            overall["total"]               += 1
            overall_lenient["correct"]     += int(correct_l)
            overall_lenient["total"]       += 1
            by_dataset[ds]["correct"]      += int(correct)
            by_dataset[ds]["total"]        += 1
            by_dataset_len[ds]["correct"]  += int(correct_l)
            by_dataset_len[ds]["total"]    += 1
            by_tasktype[tt]["correct"]     += int(correct)
            by_tasktype[tt]["total"]       += 1
            by_tasktype_len[tt]["correct"] += int(correct_l)
            by_tasktype_len[tt]["total"]   += 1
            all_samples.append({
                "dataset":          ds,
                "task_type":        tt,
                "gt":               gt,
                "pred":             pred,
                "correct":          correct,
                "correct_lenient":  correct_l,
            })

        # Update tqdm with running accuracy
        s_acc = overall["correct"] / max(1, overall["total"])
        l_acc = overall_lenient["correct"] / max(1, overall_lenient["total"])
        pbar.set_postfix(strict=f"{s_acc:.4f}", lenient=f"{l_acc:.4f}")

    def acc(s: dict) -> float:
        return s["correct"] / max(1, s["total"])

    return {
        "overall": {
            "accuracy":         acc(overall),
            "correct":          overall["correct"],
            "total":            overall["total"],
            "accuracy_lenient": acc(overall_lenient),
            "correct_lenient":  overall_lenient["correct"],
        },
        "by_dataset": {
            ds: {
                "accuracy": acc(s), "correct": s["correct"], "total": s["total"],
                "accuracy_lenient": acc(by_dataset_len[ds]),
                "correct_lenient":  by_dataset_len[ds]["correct"],
            }
            for ds, s in sorted(by_dataset.items())
        },
        "by_task_type": {
            tt: {
                "accuracy": acc(s), "correct": s["correct"], "total": s["total"],
                "accuracy_lenient": acc(by_tasktype_len[tt]),
                "correct_lenient":  by_tasktype_len[tt]["correct"],
            }
            for tt, s in sorted(by_tasktype.items())
        },
        "samples": all_samples,
    }


# ── Pretty printers ───────────────────────────────────────────────────────────

def print_results(label: str, results: dict) -> None:
    W = 72
    print()
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    ov = results["overall"]
    print(f"\n  Overall accuracy (strict)  : {ov['accuracy']:.4f}  ({ov['correct']}/{ov['total']})")
    print(f"  Overall accuracy (lenient) : {ov['accuracy_lenient']:.4f}  ({ov['correct_lenient']}/{ov['total']})")

    print(f"\n  {'By dataset:':<22}  {'strict':>10}  {'lenient':>10}")
    for ds, s in results["by_dataset"].items():
        print(f"    {ds:<20}  {s['accuracy']:>10.4f}  {s['accuracy_lenient']:>10.4f}  ({s['correct']}/{s['correct_lenient']}/{s['total']})")

    print(f"\n  {'By task type:':<22}  {'strict':>10}  {'lenient':>10}")
    for tt, s in results["by_task_type"].items():
        print(f"    {tt:<20}  {s['accuracy']:>10.4f}  {s['accuracy_lenient']:>10.4f}  ({s['correct']}/{s['correct_lenient']}/{s['total']})")

    print("=" * W)


def rescore_from_json(json_path: str) -> dict:
    """
    Load a previously-saved eval JSON and re-score all samples with both
    strict and lenient matching.  Returns a dict in the same format as
    evaluate_model() so it can be printed / saved the same way.
    """
    log.info("Re-scoring from %s", json_path)
    with open(json_path) as f:
        data = json.load(f)

    output = {"timestamp": data.get("timestamp", ""), "args": data.get("args", {})}
    output["args"]["from_json"] = json_path

    for key_prefix in ("finetuned", "pretrained"):
        samples_key = f"{key_prefix}_samples"
        if samples_key not in data:
            continue

        samples = data[samples_key]
        overall          = _empty_stats()
        overall_lenient  = _empty_stats()
        by_dataset       = defaultdict(_empty_stats)
        by_dataset_len   = defaultdict(_empty_stats)
        by_tasktype      = defaultdict(_empty_stats)
        by_tasktype_len  = defaultdict(_empty_stats)
        new_samples = []

        for s in samples:
            pred, gt, tt, ds = s["pred"], s["gt"], s["task_type"], s["dataset"]
            correct   = is_correct(pred, gt, tt)
            correct_l = is_correct_lenient(pred, gt, tt)

            overall["correct"]             += int(correct)
            overall["total"]               += 1
            overall_lenient["correct"]     += int(correct_l)
            overall_lenient["total"]       += 1
            by_dataset[ds]["correct"]      += int(correct)
            by_dataset[ds]["total"]        += 1
            by_dataset_len[ds]["correct"]  += int(correct_l)
            by_dataset_len[ds]["total"]    += 1
            by_tasktype[tt]["correct"]     += int(correct)
            by_tasktype[tt]["total"]       += 1
            by_tasktype_len[tt]["correct"] += int(correct_l)
            by_tasktype_len[tt]["total"]   += 1
            new_samples.append({
                "dataset":          ds,
                "task_type":        tt,
                "gt":               gt,
                "pred":             pred,
                "correct":          correct,
                "correct_lenient":  correct_l,
            })

        def acc(s_: dict) -> float:
            return s_["correct"] / max(1, s_["total"])

        results = {
            "overall": {
                "accuracy":         acc(overall),
                "correct":          overall["correct"],
                "total":            overall["total"],
                "accuracy_lenient": acc(overall_lenient),
                "correct_lenient":  overall_lenient["correct"],
            },
            "by_dataset": {
                ds: {
                    "accuracy": acc(s_), "correct": s_["correct"], "total": s_["total"],
                    "accuracy_lenient": acc(by_dataset_len[ds]),
                    "correct_lenient":  by_dataset_len[ds]["correct"],
                }
                for ds, s_ in sorted(by_dataset.items())
            },
            "by_task_type": {
                tt: {
                    "accuracy": acc(s_), "correct": s_["correct"], "total": s_["total"],
                    "accuracy_lenient": acc(by_tasktype_len[tt]),
                    "correct_lenient":  by_tasktype_len[tt]["correct"],
                }
                for tt, s_ in sorted(by_tasktype.items())
            },
        }

        print_results(f"RE-SCORED: {key_prefix.upper()}", results)
        output[key_prefix] = {k: v for k, v in results.items() if k != "samples"}
        output[f"{key_prefix}_samples"] = new_samples

        # Print the samples that flipped from incorrect→correct under lenient
        flipped = [s for s in new_samples if s["correct_lenient"] and not s["correct"]]
        if flipped:
            print(f"\n  Samples gained under lenient matching ({len(flipped)}):")
            for s in flipped:
                print(f"    gt={s['gt']!r:<30}  pred={s['pred']!r}")

    return output


def print_comparison(ft_results: dict, pt_results: dict) -> None:
    W = 80
    all_datasets = sorted(
        set(ft_results["by_dataset"]) | set(pt_results["by_dataset"])
    )

    print()
    print("=" * W)
    print("  COMPARISON: Fine-tuned vs Pretrained")
    print("=" * W)
    hdr = f"  {'Subset':<30} {'Fine-tuned':>11} {'Pretrained':>11} {'Delta':>9}"
    print(f"\n{hdr}")
    print("  " + "-" * (W - 2))

    def row(label, ft_acc, pt_acc):
        delta = ft_acc - pt_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<30} {ft_acc:>11.4f} {pt_acc:>11.4f} {sign}{delta:>8.4f}")

    ft_ov = ft_results["overall"]
    pt_ov = pt_results["overall"]
    row("Overall", ft_ov["accuracy"], pt_ov["accuracy"])

    print("  " + "-" * (W - 2))
    for ds in all_datasets:
        ft_s = ft_results["by_dataset"].get(ds, {"accuracy": 0.0})
        pt_s = pt_results["by_dataset"].get(ds, {"accuracy": 0.0})
        row(f"  {ds}", ft_s["accuracy"], pt_s["accuracy"])

    all_tt = sorted(
        set(ft_results["by_task_type"]) | set(pt_results["by_task_type"])
    )
    print("  " + "-" * (W - 2))
    for tt in all_tt:
        ft_s = ft_results["by_task_type"].get(tt, {"accuracy": 0.0})
        pt_s = pt_results["by_task_type"].get(tt, {"accuracy": 0.0})
        row(f"  {tt}", ft_s["accuracy"], pt_s["accuracy"])

    print("=" * W)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    log.info("Loading model from: %s", model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,          # stream shards to GPU without duplicating on CPU
        attn_implementation="sdpa",      # more memory-efficient attention than eager
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_device = next(model.get_input_embeddings().parameters()).device
    log.info("embed_device: %s", embed_device)
    return model, processor, embed_device


def load_model_from_accel_checkpoint(checkpoint_dir: str, base_model_name: str):
    """
    Load a fine-tuned model from an accelerate checkpoint.

    Supports two sub-formats (auto-detected):

    * **FSDP sharded** (train.py): ``accel_state/pytorch_model_fsdp_0/``
      containing DCP shards.  Weights are loaded via
      ``torch.distributed.checkpoint``.
    * **Non-sharded** (train_baseline.py): ``accel_state/model.safetensors``
      (or ``pytorch_model.bin``).  Weights are loaded directly with
      ``safetensors`` / ``torch.load``.

    In both cases the base pretrained model is loaded first for its
    architecture and tokenizer, then the checkpoint weights are overlaid.
    """
    accel_dir = Path(checkpoint_dir) / "accel_state"
    model_shard_dir = accel_dir / "pytorch_model_fsdp_0"
    safetensors_path = accel_dir / "model.safetensors"
    bin_path = accel_dir / "pytorch_model.bin"

    use_dcp = model_shard_dir.exists()
    use_safetensors = safetensors_path.exists()
    use_bin = bin_path.exists()

    if not (use_dcp or use_safetensors or use_bin):
        log.error(
            "No model weights found in %s — expected one of: "
            "pytorch_model_fsdp_0/ (FSDP shards), model.safetensors, "
            "or pytorch_model.bin",
            accel_dir,
        )
        sys.exit(1)

    log.info("Loading base model architecture from: %s", base_model_name)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if use_dcp:
        log.info("Loading FSDP sharded weights from: %s", model_shard_dir)
        loaded = _load_dcp_state_dict(model.state_dict(), model_shard_dir)
        model.load_state_dict(loaded, strict=False)
        model.tie_weights()
        del loaded
    elif use_safetensors:
        log.info("Loading non-sharded weights from: %s", safetensors_path)
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        del state_dict
    else:
        log.info("Loading non-sharded weights from: %s", bin_path)
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        del state_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_device = next(model.get_input_embeddings().parameters()).device
    log.info("embed_device: %s", embed_device)
    return model, processor, embed_device


def _find_key_prefix(ckpt_keys: set, model_keys: set) -> str:
    """Detect an FSDP prefix that maps *model_keys* → *ckpt_keys*.

    Returns the prefix string (possibly empty) or raises RuntimeError.
    """
    # Fast path: exact match
    if model_keys <= ckpt_keys:
        return ""
    # Try every checkpoint key against every model key to discover a prefix
    for ck in sorted(ckpt_keys):
        for mk in sorted(model_keys):
            if ck.endswith(mk):
                prefix = ck[: -len(mk)]
                # Verify prefix works for all model keys
                if all(prefix + k in ckpt_keys for k in model_keys):
                    return prefix
    raise RuntimeError(
        f"Cannot map model state-dict keys to DCP checkpoint keys.\n"
        f"  Checkpoint keys (first 5): {sorted(ckpt_keys)[:5]}\n"
        f"  Model keys     (first 5): {sorted(model_keys)[:5]}"
    )


def _load_dcp_state_dict(state_dict: dict, shard_dir: Path) -> dict:
    """Load a DCP sharded checkpoint into *state_dict* (model key names).

    Handles two common mismatches:
    * **Tied weights** (e.g. ``lm_head.weight`` shared with embeddings):
      keys present in the model but absent from the checkpoint are skipped.
    * **FSDP key prefix** (e.g. ``_fsdp_wrapped_module.``): automatically
      detected and stripped so the returned dict uses original model keys.

    Returns a dict with **model key names** and loaded tensors.
    """
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    pg_created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
        pg_created = True
    try:
        reader = dcp.FileSystemReader(str(shard_dir))
        ckpt_keys = set(reader.read_metadata().state_dict_metadata.keys())
        model_keys = set(state_dict.keys())

        prefix = _find_key_prefix(ckpt_keys, model_keys)
        if prefix:
            log.info("Detected FSDP key prefix: %r", prefix)

        # Build a dict keyed by *checkpoint* names (what DCP expects),
        # using the model's tensors as storage targets.
        ckpt_to_model = {}  # ckpt_key → model_key
        for mk in model_keys:
            ck = prefix + mk
            if ck in ckpt_keys:
                ckpt_to_model[ck] = mk
        skipped = model_keys - set(ckpt_to_model.values())
        if skipped:
            log.info("Skipping %d tied/missing keys not in DCP checkpoint: %s",
                     len(skipped), ", ".join(sorted(skipped)))

        loadable = {ck: state_dict[mk] for ck, mk in ckpt_to_model.items()}
        dcp.load(loadable, storage_reader=reader)

        # Return with original model key names
        return {mk: loadable[ck] for ck, mk in ckpt_to_model.items()}
    finally:
        if pg_created and dist.is_initialized():
            dist.destroy_process_group()


def load_projector_from_hf(checkpoint_dir: str, device) -> "PoseFeatureProjector | None":
    """Load pose projector from an HF-format checkpoint directory."""
    ckpt = Path(checkpoint_dir)
    proj_path = ckpt / "pose_projector.pt"
    config_path = ckpt / "config.json"

    if not proj_path.exists():
        log.warning("No pose_projector.pt at %s — running without projector", ckpt)
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    tokens_per_frame = cfg.get("pose_tokens_per_frame", 16)
    projector = PoseFeatureProjector(
        cfg["pose_feature_dim"], cfg["hidden_size"], tokens_per_frame=tokens_per_frame,
    )
    projector.load_state_dict(
        torch.load(proj_path, map_location="cpu", weights_only=True)
    )
    projector = projector.to(dtype=torch.bfloat16, device=device)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False
    log.info("Loaded pose projector from %s (%d → %d × %d tokens/frame)",
             proj_path, cfg["pose_feature_dim"], cfg["hidden_size"], tokens_per_frame)
    return projector


def load_projector_from_accel(checkpoint_dir: str, device) -> "PoseFeatureProjector | None":
    """Load pose projector from an accelerate FSDP checkpoint directory."""
    ckpt = Path(checkpoint_dir)
    proj_shard_dir = ckpt / "accel_state" / "pytorch_model_fsdp_1"
    state_path = ckpt / "train_state.json"

    if not proj_shard_dir.exists():
        log.warning("No pytorch_model_fsdp_1 at %s — running without projector",
                    ckpt / "accel_state")
        return None

    with open(state_path) as f:
        ts = json.load(f)

    tokens_per_frame = ts.get("pose_tokens_per_frame", 16)
    projector = PoseFeatureProjector(
        ts["feat_dim"], ts["hidden_size"], tokens_per_frame=tokens_per_frame,
    )
    loaded = _load_dcp_state_dict(projector.state_dict(), proj_shard_dir)
    projector.load_state_dict(loaded)
    projector = projector.to(dtype=torch.bfloat16, device=device)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False
    log.info("Loaded pose projector from %s (%d → %d × %d tokens/frame)",
             proj_shard_dir, ts["feat_dim"], ts["hidden_size"], tokens_per_frame)
    return projector


def _detect_checkpoint_format(checkpoint_dir: str) -> str:
    """Return 'hf' if checkpoint/model/ exists, 'accel' if accel_state/ exists."""
    ckpt = Path(checkpoint_dir)
    if (ckpt / "model").is_dir():
        return "hf"
    if (ckpt / "accel_state").is_dir():
        return "accel"
    log.error(
        "Cannot detect checkpoint format at %s — expected either a model/ "
        "subdirectory (HF format) or an accel_state/ subdirectory (accelerate format).",
        checkpoint_dir,
    )
    sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Re-score from existing JSON (no model needed) ────────────────────
    if args.from_json:
        output = rescore_from_json(args.from_json)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"eval_rescored_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Re-scored results saved to %s", out_path)
        print(f"\nRe-scored results saved to: {out_path}")
        return

    # Validate args early
    ckpt_format = None
    if args.mode in ("finetuned", "both"):
        ckpt_format = _detect_checkpoint_format(args.checkpoint)
        if ckpt_format == "accel" and not args.base_model:
            log.error(
                "Accelerate checkpoint detected at %s — "
                "--base_model is required so the model architecture can be loaded "
                "(e.g. --base_model Qwen/Qwen3.5-9B).",
                args.checkpoint,
            )
            sys.exit(1)
        log.info("Checkpoint format: %s", ckpt_format)

    if args.mode in ("pretrained", "both") and not args.pretrained_model:
        log.error("--pretrained_model is required when --mode is '%s'", args.mode)
        sys.exit(1)

    # Load eval dataset once (shared between both model runs)
    log.info("Loading eval split from multi_dataset …")
    splits = get_all_samples()
    eval_samples = splits["eval"]
    if args.max_samples is not None:
        eval_samples = eval_samples[:args.max_samples]
        log.info("Limiting to %d samples (--max_samples)", args.max_samples)
    log.info("Eval split: %d samples (pose features extracted online per frame)", len(eval_samples))

    output = {"timestamp": timestamp, "args": vars(args)}

    def run_one(model_path: str, label: str, *,
                accel_ckpt: str = None, base_model: str = None,
                checkpoint_dir: str = None, ckpt_fmt: str = None) -> dict:
        """Load a model (and optionally its projector) and evaluate it.

        For HF checkpoints, pass model_path pointing to the saved model dir.
        For accel checkpoints, pass accel_ckpt (the resume_ckpt dir) and
        base_model (the pretrained model name for architecture).

        When *checkpoint_dir* and *ckpt_fmt* are given, the pose projector is
        loaded from the checkpoint and used during generation.
        """
        if accel_ckpt is not None:
            model, processor, embed_device = load_model_from_accel_checkpoint(
                accel_ckpt, base_model,
            )
        else:
            model, processor, embed_device = load_model(model_path)

        # Load projector for fine-tuned evaluation
        projector = None
        if checkpoint_dir is not None and ckpt_fmt is not None:
            if ckpt_fmt == "accel":
                projector = load_projector_from_accel(checkpoint_dir, embed_device)
            else:
                projector = load_projector_from_hf(checkpoint_dir, embed_device)
            if projector is not None:
                log.info("Pose projector loaded — will be used for generation")

        ds = MultiVideoDataset(
            eval_samples,
            fps=args.fps,
            max_frames=args.max_frames,
        )

        if projector is not None:
            collate_fn = lambda b: _collate_gen(
                b, processor, embed_device, args.pose_tokens_per_frame,
            )
        else:
            collate_fn = lambda b: _collate_gen_vanilla(b, processor, embed_device)

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        log.info("Evaluating %s on %d samples …", label, len(ds))
        results = evaluate_model(model, processor, loader, args.max_new_tokens,
                                 projector=projector)
        print_results(label, results)

        # Free GPU memory before potentially loading the next model
        del model, projector
        torch.cuda.empty_cache()

        return results

    ft_results = pt_results = None

    if args.mode in ("finetuned", "both"):
        if ckpt_format == "accel":
            ft_results = run_one(
                None, "FINE-TUNED MODEL",
                accel_ckpt=args.checkpoint, base_model=args.base_model,
                checkpoint_dir=args.checkpoint, ckpt_fmt=ckpt_format,
            )
        else:
            ft_model_path = str(Path(args.checkpoint) / "model")
            ft_results = run_one(
                ft_model_path, "FINE-TUNED MODEL",
                checkpoint_dir=args.checkpoint, ckpt_fmt=ckpt_format,
            )
        output["finetuned"] = {k: v for k, v in ft_results.items() if k != "samples"}
        output["finetuned_samples"] = ft_results["samples"]

    if args.mode in ("pretrained", "both"):
        pt_results = run_one(args.pretrained_model, "PRETRAINED MODEL (baseline)")
        output["pretrained"] = {k: v for k, v in pt_results.items() if k != "samples"}
        output["pretrained_samples"] = pt_results["samples"]

    if args.mode == "both":
        print_comparison(ft_results, pt_results)
        all_datasets = sorted(
            set(ft_results["by_dataset"]) | set(pt_results["by_dataset"])
        )
        output["comparison"] = {
            "overall": {
                "finetuned_acc":  ft_results["overall"]["accuracy"],
                "pretrained_acc": pt_results["overall"]["accuracy"],
                "delta":          ft_results["overall"]["accuracy"] - pt_results["overall"]["accuracy"],
            },
            **{
                ds: {
                    "finetuned_acc":  ft_results["by_dataset"].get(ds, {"accuracy": 0.0})["accuracy"],
                    "pretrained_acc": pt_results["by_dataset"].get(ds, {"accuracy": 0.0})["accuracy"],
                    "delta":          ft_results["by_dataset"].get(ds, {"accuracy": 0.0})["accuracy"]
                                    - pt_results["by_dataset"].get(ds, {"accuracy": 0.0})["accuracy"],
                }
                for ds in all_datasets
            },
        }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eval_{args.mode}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", out_path)
    print(f"\nResults saved to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen checkpoints on the QVID+Kinetics+HMDB51 eval split"
    )
    p.add_argument(
        "--from_json", default=None, metavar="PATH",
        help="Re-score an existing eval JSON with strict + lenient metrics "
             "(skips model loading entirely)",
    )
    p.add_argument(
        "--mode", default="finetuned",
        choices=["finetuned", "pretrained", "both"],
        help="Which model(s) to evaluate (default: finetuned)",
    )
    p.add_argument(
        "--checkpoint",
        default="/orcd/compute/ppliang/001/qwen_multi/resume_ckpt",
        help="Checkpoint dir — either an HF checkpoint (with model/ subdir) "
             "or an accelerate resume checkpoint (with accel_state/ subdir). "
             "Format is auto-detected. (default: %(default)s)",
    )
    p.add_argument(
        "--base_model",
        default="Qwen/Qwen3.5-9B",
        help="HF model name or local path for the base pretrained model "
             "architecture. Required when --checkpoint is in accelerate format "
             "(e.g. Qwen/Qwen3.5-9B).",
    )
    p.add_argument(
        "--pretrained_model",
        default="Qwen/Qwen3.5-9B",
        help="HF model name or local path for the pretrained baseline "
             "(required when --mode is pretrained or both)",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Max tokens to generate per sample (default: 32)",
    )
    p.add_argument("--max_samples",   type=int,   default=None,
                   help="Limit eval to this many samples (e.g. 100 for a quick run)")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--fps",           type=float, default=2.0)
    p.add_argument("--max_frames",    type=int,   default=32)
    p.add_argument("--pose_tokens_per_frame", type=int, default=16,
                   help="Pose tokens emitted per sampled video frame (default: 16).")
    p.add_argument(
        "--output_dir", default="/orcd/compute/ppliang/001/qwen_multi/results",
        help="Directory for timestamped JSON output (default: ./eval_results)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
