#!/usr/bin/env python3
"""
evaluate_multidataset_no_video.py — Evaluate a Qwen checkpoint on the eval split
of QVID + Kinetics-400 + HMDB51 using ONLY pose tokens (no video frames).

The pose projector trained via train.py produces ``pose_sample_n`` embeddings
that are injected at the prompt/answer boundary. Everything about the run
mirrors evaluate_multidataset.py except that no ``{"type": "image"}`` entries
are added to the chat template, so the processor never builds
pixel_values / image_grid_thw / pixel_values_videos / video_grid_thw.
The hypothesis is that the pose tokens alone carry most of the information
needed for these tasks.

Modes
-----
  finetuned  : load the fine-tuned backbone from --checkpoint (with projector)
  pretrained : load the original model from --pretrained_model (with projector
               from --checkpoint; if not provided, projector is skipped)
  both       : run both and print a side-by-side comparison

Usage
-----
  # Fine-tuned model + projector, no video tokens
  python evaluate_multidataset_no_video.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_multi/best

  # Accelerate resume checkpoint
  python evaluate_multidataset_no_video.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_multi/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B

  # Side-by-side vs pretrained baseline (baseline gets pose tokens too, using
  # the same trained projector, so the only difference is the backbone)
  python evaluate_multidataset_no_video.py \\
      --mode both \\
      --checkpoint /orcd/compute/ppliang/001/qwen_multi/best \\
      --pretrained_model Qwen/Qwen3.5-9B
"""

import argparse
import json
import logging
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
from evaluate_multidataset import (
    is_correct,
    is_correct_lenient,
    print_results,
    print_comparison,
    load_model,
    load_model_from_accel_checkpoint,
    load_projector_from_hf,
    load_projector_from_accel,
    _detect_checkpoint_format,
    _empty_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Collate: pose-only (no video frames) ──────────────────────────────────────

def _collate_gen_pose_only(batch: List[dict], processor,
                            pose_sample_n: int = 16) -> dict:
    """Build prompt-only inputs with pose placeholders appended at the end.

    Crucially, no ``{"type": "image"}`` entries are added to the chat template
    and no images are passed to the processor — so the processor returns pure
    text inputs without any visual token placeholders or pixel tensors.

    Returns CPU tensors; the eval loop is responsible for moving to GPU.  This
    keeps the collate fork-safe so ``num_workers > 0`` doesn't trip CUDA
    re-initialization in workers.
    """
    texts = []
    for ex in batch:
        content = [{"type": "text", "text": ex["question"]}]
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    inputs = processor(text=texts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    B, orig_len = input_ids.shape
    n_pose = pose_sample_n
    new_len = orig_len + n_pose

    new_ids  = torch.full((B, new_len), pad_id, dtype=input_ids.dtype)
    new_mask = torch.zeros((B, new_len), dtype=inputs["attention_mask"].dtype)

    pose_positions = []
    for i in range(B):
        nonpad = (input_ids[i] != pad_id).sum().item()
        ap = nonpad  # insert at end of prompt
        ps, pe = ap, ap + n_pose

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
        "pose_feat":      torch.stack([ex["pose_feat"] for ex in batch]),
    }


# ── Dataset that loads pose only (no video decoding) ─────────────────────────

class PoseOnlyDataset(MultiVideoDataset):
    """Identical to MultiVideoDataset but skips video-frame decoding so the
    DataLoader doesn't pay the cost of reading frames that will never be used."""

    def __getitem__(self, idx: int) -> dict:
        s = self.examples[idx]
        from train import _load_pose_features
        pose_feat = _load_pose_features(s["pose_path"], self.pose_sample_n)
        return {
            "pose_feat":  pose_feat,
            "question":   s["question"],
            "answer":     s["answer"],
            "dataset":    s["dataset"],
            "task_type":  s["task_type"],
        }


# ── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_with_projector_no_video(model, projector, batch,
                                       max_new_tokens, pad_id, eos_id):
    """Project pose features, inject via hook, and generate.  No video tokens."""
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
def evaluate_model_no_video(model, processor, loader, max_new_tokens: int,
                             projector, embed_device) -> dict:
    """Generate pose-only responses and score against ground truth.

    A projector is **required** — without video tokens, the pretrained model
    cannot see the video at all, so running without a projector is a pure
    text-only baseline (still supported: pass ``projector=None``).
    """
    from collections import defaultdict

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

    pbar = tqdm(loader, desc="Generating (no video)", dynamic_ncols=True)
    for batch in pbar:
        answers    = batch.pop("answers")
        datasets   = batch.pop("datasets")
        task_types = batch.pop("task_types")

        batch["input_ids"]      = batch["input_ids"].to(embed_device)
        batch["attention_mask"] = batch["attention_mask"].to(embed_device)
        if "pose_feat" in batch:
            batch["pose_feat"]  = batch["pose_feat"].to(embed_device)

        if projector is not None:
            new_ids = _generate_with_projector_no_video(
                model, projector, batch, max_new_tokens, pad_id, eos_id,
            )
        else:
            # Text-only control: drop pose too and generate from prompt alone
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

        preds = [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in new_ids
        ]

        for pred, gt, ds, tt in zip(preds, answers, datasets, task_types):
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
            all_samples.append({
                "dataset":          ds,
                "task_type":        tt,
                "gt":               gt,
                "pred":             pred,
                "correct":          correct,
                "correct_lenient":  correct_l,
            })

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_format = None
    if args.mode in ("finetuned", "both"):
        ckpt_format = _detect_checkpoint_format(args.checkpoint)
        if ckpt_format == "accel" and not args.base_model:
            log.error(
                "Accelerate checkpoint detected at %s — --base_model is required.",
                args.checkpoint,
            )
            raise SystemExit(1)
        log.info("Checkpoint format: %s", ckpt_format)

    if args.mode in ("pretrained", "both") and not args.pretrained_model:
        log.error("--pretrained_model is required when --mode is '%s'", args.mode)
        raise SystemExit(1)

    log.info("Loading eval split from multi_dataset …")
    splits = get_all_samples()
    eval_samples = splits["eval"]
    if args.max_samples is not None:
        eval_samples = eval_samples[:args.max_samples]
        log.info("Limiting to %d samples (--max_samples)", args.max_samples)
    log.info("Eval split: %d samples", len(eval_samples))

    output = {"timestamp": timestamp, "args": vars(args)}

    def run_one(label: str, *, accel_ckpt=None, base_model=None, model_path=None,
                checkpoint_dir=None, ckpt_fmt=None) -> dict:
        if accel_ckpt is not None:
            model, processor, embed_device = load_model_from_accel_checkpoint(
                accel_ckpt, base_model,
            )
        else:
            model, processor, embed_device = load_model(model_path)

        projector = None
        if checkpoint_dir is not None and ckpt_fmt is not None:
            if ckpt_fmt == "accel":
                projector = load_projector_from_accel(checkpoint_dir, embed_device)
            else:
                projector = load_projector_from_hf(checkpoint_dir, embed_device)
            if projector is not None:
                log.info("Pose projector loaded — will be used for generation")

        if projector is None and not args.allow_no_projector:
            raise RuntimeError(
                "No projector could be loaded, and --allow_no_projector was not "
                "set.  With no video tokens and no projector the model has no "
                "access to the video at all — aborting."
            )

        ds = PoseOnlyDataset(
            eval_samples,
            fps=args.fps,
            max_frames=args.max_frames,
            pose_sample_n=args.pose_sample_n,
            skip_missing_pose=True,
        )

        collate_fn = lambda b: _collate_gen_pose_only(
            b, processor, args.pose_sample_n,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        log.info("Evaluating %s on %d samples (pose-only, no video) …", label, len(ds))
        results = evaluate_model_no_video(
            model, processor, loader, args.max_new_tokens,
            projector=projector, embed_device=embed_device,
        )
        print_results(label, results)

        del model, projector
        torch.cuda.empty_cache()
        return results

    ft_results = pt_results = None

    if args.mode in ("finetuned", "both"):
        if ckpt_format == "accel":
            ft_results = run_one(
                "FINE-TUNED MODEL (pose-only)",
                accel_ckpt=args.checkpoint, base_model=args.base_model,
                checkpoint_dir=args.checkpoint, ckpt_fmt=ckpt_format,
            )
        else:
            ft_model_path = str(Path(args.checkpoint) / "model")
            ft_results = run_one(
                "FINE-TUNED MODEL (pose-only)",
                model_path=ft_model_path,
                checkpoint_dir=args.checkpoint, ckpt_fmt=ckpt_format,
            )
        output["finetuned"] = {k: v for k, v in ft_results.items() if k != "samples"}
        output["finetuned_samples"] = ft_results["samples"]

    if args.mode in ("pretrained", "both"):
        pt_results = run_one(
            "PRETRAINED MODEL (pose-only baseline)",
            model_path=args.pretrained_model,
            checkpoint_dir=args.checkpoint if args.use_projector_for_pretrained else None,
            ckpt_fmt=ckpt_format if args.use_projector_for_pretrained else None,
        )
        output["pretrained"] = {k: v for k, v in pt_results.items() if k != "samples"}
        output["pretrained_samples"] = pt_results["samples"]

    if args.mode == "both":
        print_comparison(ft_results, pt_results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eval_no_video_{args.mode}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", out_path)
    print(f"\nResults saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a pose-projector Qwen checkpoint on QVID+Kinetics+HMDB51 "
                    "using pose tokens only (no video frames)."
    )
    p.add_argument(
        "--mode", default="finetuned",
        choices=["finetuned", "pretrained", "both"],
        help="Which model(s) to evaluate (default: finetuned)",
    )
    p.add_argument(
        "--checkpoint",
        default="/orcd/compute/ppliang/001/qwen_multi/best",
        help="Checkpoint dir — either an HF checkpoint (with model/ subdir) "
             "or an accelerate resume checkpoint (with accel_state/ subdir).",
    )
    p.add_argument(
        "--base_model",
        default="Qwen/Qwen3.5-9B",
        help="Required when --checkpoint is in accelerate format.",
    )
    p.add_argument(
        "--pretrained_model",
        default="Qwen/Qwen3.5-9B",
        help="HF model name or local path for the pretrained baseline.",
    )
    p.add_argument(
        "--use_projector_for_pretrained", action="store_true", default=True,
        help="Also use the trained projector for the pretrained baseline so "
             "only the backbone differs (default: on).",
    )
    p.add_argument(
        "--no_projector_for_pretrained", dest="use_projector_for_pretrained",
        action="store_false",
    )
    p.add_argument(
        "--allow_no_projector", action="store_true", default=False,
        help="Skip the safety check that aborts when no projector is found "
             "(e.g. to measure the prompt-only accuracy of the backbone).",
    )
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--batch_size",     type=int, default=1)
    p.add_argument("--num_workers",    type=int, default=2)
    p.add_argument("--fps",            type=float, default=2.0,
                   help="Unused here (kept for API parity with train.py).")
    p.add_argument("--max_frames",     type=int, default=32,
                   help="Unused here (kept for API parity with train.py).")
    p.add_argument("--pose_sample_n",  type=int, default=16)
    p.add_argument(
        "--output_dir",
        default="/orcd/compute/ppliang/001/qwen_multi/results",
        help="Directory for timestamped JSON output.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
