#!/usr/bin/env python3
"""
evaluate_multidataset.py — Evaluate a Qwen checkpoint on the eval split of
QVID + Kinetics-400 + HMDB51 using full auto-regressive generation.

Modes
-----
  finetuned  : load the fine-tuned backbone from --checkpoint/model/
  pretrained : load the original model from --pretrained_model (no fine-tuning)
  both       : run both and print a side-by-side comparison

The pose projector is NOT used during generation — it was an auxiliary training
head only.  Both modes evaluate the backbone's generation quality directly.

Results are printed to stdout and saved to --output_dir/eval_<mode>_<timestamp>.json.

Usage
-----
  # Fine-tuned only (default)
  python evaluate_multidataset.py \\
      --checkpoint /home/ixzhu/orcd/scratch/qwen_pose/resume_ckpt

  # Pretrained baseline only
  python evaluate_multidataset.py \\
      --mode pretrained \\
      --pretrained_model Qwen/Qwen3.5-VL-7B-Instruct

  # Side-by-side comparison
  python evaluate_multidataset.py \\
      --mode both \\
      --checkpoint /home/ixzhu/orcd/scratch/qwen_pose/resume_ckpt \\
      --pretrained_model Qwen/Qwen3.5-VL-7B-Instruct
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from train import MultiVideoDataset
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


# ── Collate for generation ────────────────────────────────────────────────────

def _collate_gen(batch: List[dict], processor, embed_device) -> dict:
    """
    Build processor inputs for the prompt only (no answer tokens in input).
    Ground-truth answers and metadata are passed through as plain lists.
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
            )
        )
        all_images.append(ex["pil_images"])

    inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )
    return {
        **{k: v.to(embed_device) for k, v in inputs.items()},
        "answers":    [ex["answer"]    for ex in batch],
        "datasets":   [ex["dataset"]   for ex in batch],
        "task_types": [ex["task_type"] for ex in batch],
    }


# ── Evaluation loop ───────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {"correct": 0, "total": 0}


@torch.no_grad()
def evaluate_model(model, processor, loader, max_new_tokens: int) -> dict:
    """
    Generate responses for every batch and compare against ground truth.
    Returns a dict with per-dataset, per-task-type, and overall accuracy,
    plus a full list of per-sample results for offline analysis.
    """
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    overall     = _empty_stats()
    by_dataset  = defaultdict(_empty_stats)
    by_tasktype = defaultdict(_empty_stats)
    all_samples = []

    model.eval()

    for batch in tqdm(loader, desc="Generating", dynamic_ncols=True):
        answers    = batch.pop("answers")
        datasets   = batch.pop("datasets")
        task_types = batch.pop("task_types")

        prompt_len = batch["input_ids"].shape[1]

        # The checkpoint is Qwen3_5ForCausalLM (text-only backbone).  The
        # processor still returns pixel_values / image_grid_thw / mm_token_type_ids
        # but the text model ignores them — visual information is carried only
        # through the image-placeholder token IDs already in input_ids.
        # Passing those tensors to generate() causes a validation error, so we
        # strip them here.
        _VISION_KEYS = {
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "mm_token_type_ids",
        }
        gen_kwargs = {
            k: v for k, v in batch.items()
            if k not in _VISION_KEYS and v is not None
        }

        gen_ids = model.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        # Decode only the newly generated tokens
        new_ids = gen_ids[:, prompt_len:]
        preds = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in new_ids
        ]

        for pred, gt, ds, tt in zip(preds, answers, datasets, task_types):
            correct = is_correct(pred, gt, tt)
            overall["correct"]         += int(correct)
            overall["total"]           += 1
            by_dataset[ds]["correct"]  += int(correct)
            by_dataset[ds]["total"]    += 1
            by_tasktype[tt]["correct"] += int(correct)
            by_tasktype[tt]["total"]   += 1
            all_samples.append({
                "dataset":   ds,
                "task_type": tt,
                "gt":        gt,
                "pred":      pred,
                "correct":   correct,
            })

    def acc(s: dict) -> float:
        return s["correct"] / max(1, s["total"])

    return {
        "overall": {
            "accuracy": acc(overall),
            "correct":  overall["correct"],
            "total":    overall["total"],
        },
        "by_dataset": {
            ds: {"accuracy": acc(s), "correct": s["correct"], "total": s["total"]}
            for ds, s in sorted(by_dataset.items())
        },
        "by_task_type": {
            tt: {"accuracy": acc(s), "correct": s["correct"], "total": s["total"]}
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
    print(f"\n  Overall accuracy : {ov['accuracy']:.4f}  ({ov['correct']}/{ov['total']})")

    print("\n  By dataset:")
    for ds, s in results["by_dataset"].items():
        print(f"    {ds:<12}  acc={s['accuracy']:.4f}  ({s['correct']}/{s['total']})")

    print("\n  By task type:")
    for tt, s in results["by_task_type"].items():
        print(f"    {tt:<22}  acc={s['accuracy']:.4f}  ({s['correct']}/{s['total']})")

    print("=" * W)


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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    embed_device = next(model.model.embed_tokens.parameters()).device
    log.info("embed_device: %s", embed_device)
    return model, processor, embed_device


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Validate args early
    if args.mode in ("finetuned", "both"):
        ft_model_path = str(Path(args.checkpoint) / "model")
        if not Path(ft_model_path).exists():
            log.error("Fine-tuned model not found at %s", ft_model_path)
            sys.exit(1)

    if args.mode in ("pretrained", "both") and not args.pretrained_model:
        log.error("--pretrained_model is required when --mode is '%s'", args.mode)
        sys.exit(1)

    # Load eval dataset once (shared between both model runs)
    log.info("Loading eval split from multi_dataset …")
    splits = get_all_samples()
    eval_samples = splits["eval"]
    log.info("Eval split: %d samples (after pose-file filtering happens in Dataset)", len(eval_samples))

    output = {"timestamp": timestamp, "args": vars(args)}

    def run_one(model_path: str, label: str) -> dict:
        model, processor, embed_device = load_model(model_path)

        ds = MultiVideoDataset(
            eval_samples,
            fps=args.fps,
            max_frames=args.max_frames,
            pose_sample_n=args.pose_sample_n,
            skip_missing_pose=True,  # consistent with training; ensures same samples across modes
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: _collate_gen(b, processor, embed_device),
        )

        log.info("Evaluating %s on %d samples …", label, len(ds))
        results = evaluate_model(model, processor, loader, args.max_new_tokens)
        print_results(label, results)

        # Free GPU memory before potentially loading the next model
        del model
        torch.cuda.empty_cache()

        return results

    ft_results = pt_results = None

    if args.mode in ("finetuned", "both"):
        ft_results = run_one(ft_model_path, "FINE-TUNED MODEL")
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
        "--mode", default="finetuned",
        choices=["finetuned", "pretrained", "both"],
        help="Which model(s) to evaluate (default: finetuned)",
    )
    p.add_argument(
        "--checkpoint",
        default="/home/ixzhu/orcd/scratch/qwen_pose/resume_ckpt",
        help="Resume checkpoint dir containing model/ subdir (default: %(default)s)",
    )
    p.add_argument(
        "--pretrained_model",
        default=None,
        help="HF model name or local path for the pretrained baseline "
             "(required when --mode is pretrained or both)",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Max tokens to generate per sample (default: 32)",
    )
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--fps",           type=float, default=2.0)
    p.add_argument("--max_frames",    type=int,   default=32)
    p.add_argument("--pose_sample_n", type=int,   default=16)
    p.add_argument(
        "--output_dir", default="./eval_results",
        help="Directory for timestamped JSON output (default: ./eval_results)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
