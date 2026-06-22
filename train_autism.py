#!/usr/bin/env python3
"""
train_autism.py — Fine-tune on the av-asd autism behaviour dataset.

The av-asd splits (441 train / 153 val / 143 eval) live in
``/orcd/compute/ppliang/001/jadali85/autism_datasets/av-asd``. The task is
multi-label classification over 10 behaviours; the answer string is the
comma-separated list of behaviour names. Audio placeholders are stripped from
the prompts because this model is video + pose only.

Two modes:
  --mode pose      → run train.py     (full pose pipeline)
  --mode no_pose   → run train_baseline.py  (Qwen video only, no pose tokens)

Pre-extract pose features first if using --mode pose:
  python extract_poses_per_frame.py --dataset av_asd --workers 4 \
                                    --fps 1.0 --max_frames 16

Examples:
  # Fine-tune with pose tokens, picking up your multi-dataset pretrained ckpt:
  python train_autism.py --mode pose

  # Train baseline (no pose) from base Qwen:
  python train_autism.py --mode no_pose \\
      --model_name Qwen/Qwen3.5-VL-7B-Instruct

Implementation: this is a thin wrapper that monkey-patches
``multi_dataset.get_all_samples`` so the inner train script (train.py or
train_baseline.py) sees only the av-asd splits, then dispatches into its
``train()`` function unchanged. Autism-tuned defaults are injected before
the inner argparse runs; any flag you pass on the CLI overrides them.
"""

import argparse
import sys


def _autism_defaults(mode: str) -> list:
    """CLI-style defaults layered before user-supplied args. User wins on conflict."""
    return [
        # 441 train samples × 8 epochs / (batch 1 × grad_accum 8) ≈ 440 opt steps.
        "--num_epochs", "8",
        # Gentler LR on the pretrained backbone; small dataset.
        "--backbone_lr", "5e-6",
        "--lr", "1e-4",
        # Validate / checkpoint more often on this short run.
        "--val_steps", "20",
        "--save_steps", "20",
        # Multi-label answers can be ~25 tokens for the busy clips.
        "--val_max_new_tokens", "64",
        # Mode-aware output / run identifier so pose and no-pose checkpoints
        # don't collide.
        "--output_dir", f"/orcd/compute/ppliang/001/qwen_autism_{mode}",
        "--wandb_project", "qwen-autism",
        "--wandb_run_name", f"autism-{mode}",
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["pose", "no_pose"],
        required=True,
        help="`pose` → train.py (full pose pipeline). "
             "`no_pose` → train_baseline.py (video only, no projector).",
    )
    p.add_argument(
        "--variant",
        choices=["multilabel"],
        default="multilabel",
        help="Which av-asd answer format to fit. Only `multilabel` is wired "
             "right now (plain comma-separated labels).",
    )
    args, extra_args = p.parse_known_args()

    # Patch multi_dataset.get_all_samples BEFORE the inner train script imports
    # it, so train.py / train_baseline.py see only the autism splits.
    # We concatenate the two autism datasets — av-asd (open-ended multi-label
    # behaviour identification) and scrape_asd (binary Typical / Red Flags
    # multiple-choice) — into one combined train/val/eval set.
    import multi_dataset
    av_asd_splits   = multi_dataset.get_av_asd_samples(variant=args.variant)
    scrape_asd_splits = multi_dataset.get_scrape_asd_samples()
    splits = {
        split: av_asd_splits[split] + scrape_asd_splits[split]
        for split in ("train", "val", "eval")
    }

    def _counts(s):
        return {
            "av_asd":     sum(1 for x in s if x["dataset"] == "av_asd"),
            "scrape_asd": sum(1 for x in s if x["dataset"] == "scrape_asd"),
        }
    for split_name, samples in splits.items():
        print(
            f"[train_autism] {split_name}: total={len(samples)}  {_counts(samples)}",
            flush=True,
        )

    multi_dataset.get_all_samples = lambda *a, **k: splits

    # Build the argv the inner train script will see. Defaults first (so the
    # user's CLI args override them at argparse time), then everything the
    # user passed.
    sys.argv = [sys.argv[0]] + _autism_defaults(args.mode) + extra_args

    if args.mode == "pose":
        from train import train, parse_args as inner_parse
    else:
        from train_baseline import train, parse_args as inner_parse

    train(inner_parse())


if __name__ == "__main__":
    main()
