#!/usr/bin/env python3
"""
evaluate_autism.py — Evaluate a Qwen checkpoint on the autism datasets
(av-asd + scrape_asd) using full auto-regressive generation.

Thin wrapper around ``evaluate_multidataset.main``: monkey-patches
``multi_dataset.get_all_samples`` to return the combined autism splits, then
delegates to the existing evaluator unchanged. Per-dataset and per-task-type
metrics in the output JSON will report ``av_asd`` and ``scrape_asd`` rows
separately.

Examples
--------
  # Fine-tuned pose-mode checkpoint (resume_ckpt dir, accel format)
  python evaluate_autism.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_autism_pose/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B

  # No-pose baseline checkpoint
  python evaluate_autism.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_autism_no_pose/resume_ckpt \\
      --base_model Qwen/Qwen3.5-VL-7B-Instruct

  # Pretrained baseline only
  python evaluate_autism.py \\
      --mode pretrained \\
      --pretrained_model Qwen/Qwen3.5-VL-7B-Instruct

Notes
-----
* av-asd answers are short multi-label strings (~25 tokens for busy clips), so
  the default ``--max_new_tokens`` is bumped from 32 → 64.
* Output dir defaults to ``/orcd/compute/ppliang/001/qwen_autism/results``
  so autism evals don't clobber the multi-dataset results dir.
* All other flags from ``evaluate_multidataset.py`` (``--mode``, ``--checkpoint``,
  ``--base_model``, ``--pretrained_model``, ``--max_samples``, ``--batch_size``,
  ``--fps``, ``--max_frames``, ``--pose_tokens_per_frame``, ``--from_json``,
  etc.) pass through unchanged — supply them on the CLI as you would there.
"""

import argparse
import sys


# Autism-specific defaults layered before user-supplied args. User overrides win.
AUTISM_EVAL_DEFAULTS = [
    # Multi-label av-asd answers can be ~25 tokens; default 32 truncates them.
    "--max_new_tokens", "64",
    # Don't write into the multi-dataset results dir.
    "--output_dir", "/orcd/compute/ppliang/001/qwen_autism/results",
]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variant",
        choices=["multilabel"],
        default="multilabel",
        help="Which av-asd answer format to evaluate. Only `multilabel` is "
             "wired right now (plain comma-separated labels). scrape_asd is "
             "always included; this flag only selects the av-asd variant.",
    )
    autism_args, extra_args = p.parse_known_args()

    # Build the combined autism eval splits and patch get_all_samples BEFORE
    # evaluate_multidataset imports it. We patch all three splits even though
    # the evaluator only reads ``splits["eval"]`` — keeps the structure
    # symmetric with train_autism.py.
    import multi_dataset
    av_asd_splits     = multi_dataset.get_av_asd_samples(variant=autism_args.variant)
    scrape_asd_splits = multi_dataset.get_scrape_asd_samples()
    splits = {
        split: av_asd_splits[split] + scrape_asd_splits[split]
        for split in ("train", "val", "eval")
    }

    def _counts(samples):
        return {
            "av_asd":     sum(1 for x in samples if x["dataset"] == "av_asd"),
            "scrape_asd": sum(1 for x in samples if x["dataset"] == "scrape_asd"),
        }
    print(
        f"[evaluate_autism] eval split: total={len(splits['eval'])}  "
        f"{_counts(splits['eval'])}",
        flush=True,
    )

    multi_dataset.get_all_samples = lambda *a, **k: splits

    # Forward to evaluate_multidataset.main with autism defaults layered in.
    sys.argv = [sys.argv[0]] + AUTISM_EVAL_DEFAULTS + extra_args

    from evaluate_multidataset import main as inner_main, parse_args as inner_parse
    inner_main(inner_parse())


if __name__ == "__main__":
    main()
