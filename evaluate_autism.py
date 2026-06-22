#!/usr/bin/env python3
"""
evaluate_autism.py — Evaluate a Qwen checkpoint on the autism datasets
(av-asd + scrape_asd) using full auto-regressive generation.

Thin wrapper around ``evaluate_multidataset.main``: monkey-patches
``multi_dataset.get_all_samples`` to return the combined autism splits, then
delegates to the existing evaluator unchanged. Per-dataset and per-task-type
metrics in the output JSON will report ``av_asd`` and ``scrape_asd`` rows
separately.

Supports both pose-mode checkpoints (from ``train_avasd.py``) and no-pose
checkpoints (from ``train_avasd_no_pose.py`` / ``train_baseline.py``). For
no-pose models, pass ``--no_pose`` so MediaPipe pose extraction is skipped
during eval — the no-pose path is also auto-detected when the underlying
projector loaders return None.

Examples
--------
  # Fine-tuned pose-mode checkpoint (resume_ckpt dir, accel format)
  python evaluate_autism.py \\
      --checkpoint /orcd/compute/ppliang/001/qwen_autism_pose/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B

  # No-pose checkpoint trained by train_avasd_no_pose.py / train_baseline.py
  python evaluate_autism.py \\
      --no_pose \\
      --checkpoint /orcd/compute/ppliang/001/qwen_autism_no_pose/resume_ckpt \\
      --base_model Qwen/Qwen3.5-9B

  # Pretrained baseline only
  python evaluate_autism.py \\
      --no_pose \\
      --mode pretrained \\
      --pretrained_model Qwen/Qwen3.5-VL-7B-Instruct

Notes
-----
* av-asd answers are short multi-label strings (~25 tokens for busy clips), so
  the default ``--max_new_tokens`` is bumped from 32 → 64.
* Output dir defaults to ``/orcd/compute/ppliang/001/qwen_autism/results``
  so autism evals don't clobber the multi-dataset results dir.
* ``--no_pose`` swaps in a video-only dataset (no MediaPipe pose extraction)
  and is implied when the checkpoint contains no projector weights.
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
    p.add_argument(
        "--filter_oversized_videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop eval samples whose video would exceed --max_visual_tokens. "
             "Mirrors the same filter used in train_avasd.py to avoid OOM on "
             "high-resolution samples (the lm_head's (B, seq_len, vocab) "
             "logits dominate per-sample memory). Default on; pass "
             "--no-filter_oversized_videos to evaluate every sample.",
    )
    p.add_argument(
        "--max_visual_tokens",
        type=int,
        default=40000,
        help="Visual-token budget for the filter: drop videos where "
             "ceil(W/28)*ceil(H/28)*max_frames exceeds this. Default 40000 "
             "(~1080p at 16 frames, with headroom for text/pose tokens). "
             "Ignored when --no-filter_oversized_videos is set.",
    )
    p.add_argument(
        "--no_pose",
        action="store_true",
        default=False,
        help="Evaluate a model that has no pose projector (e.g. trained by "
             "train_avasd_no_pose.py or train_baseline.py). Skips MediaPipe "
             "pose extraction in the dataset and forces the vanilla "
             "(no-pose-tokens) collate inside evaluate_multidataset. The "
             "no-pose collate also kicks in automatically when the checkpoint "
             "loader finds no projector weights; this flag just saves the "
             "wasted pose-extraction work in that case.",
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
        f"[evaluate_autism] eval split (pre-filter): total={len(splits['eval'])}  "
        f"{_counts(splits['eval'])}",
        flush=True,
    )

    # ── Visual-token budget filter ──────────────────────────────────────────
    # Mirror train_avasd.py's pre-filter so eval doesn't OOM on the same 4K /
    # ultra-high-res samples that training already skips.  Done at the splits
    # level (before evaluate_multidataset's MultiVideoDataset sees them) so
    # we don't have to modify evaluate_multidataset.py or train.py's dataset.
    if autism_args.filter_oversized_videos and autism_args.max_visual_tokens > 0:
        from tqdm import tqdm as _tqdm
        from train_avasd import _is_video_readable as _avasd_is_readable

        # Use the same max_frames the evaluator will actually feed the model.
        # Peek at extra_args (forwarded to evaluate_multidataset, which
        # defaults to 32).  Avoids "filter says fits, eval OOMs" mismatch.
        max_frames_for_budget = 32
        for i, tok in enumerate(extra_args):
            if tok == "--max_frames" and i + 1 < len(extra_args):
                try:
                    max_frames_for_budget = int(extra_args[i + 1])
                except ValueError:
                    pass
                break
            if tok.startswith("--max_frames="):
                try:
                    max_frames_for_budget = int(tok.split("=", 1)[1])
                except ValueError:
                    pass
                break

        def _filter(samples, label):
            kept = []
            for s in _tqdm(samples, desc=f"Filtering {label} videos",
                           file=sys.stdout):
                if _avasd_is_readable(
                    s["video_path"],
                    max_visual_tokens=autism_args.max_visual_tokens,
                    max_frames_for_budget=max_frames_for_budget,
                ):
                    kept.append(s)
            n_drop = len(samples) - len(kept)
            if n_drop:
                print(
                    f"[evaluate_autism] {label}: dropped {n_drop} / "
                    f"{len(samples)} videos exceeding "
                    f"{autism_args.max_visual_tokens} visual tokens "
                    f"(max_frames_for_budget={max_frames_for_budget})",
                    flush=True,
                )
            return kept

        splits = {k: _filter(v, k) for k, v in splits.items()}
        print(
            f"[evaluate_autism] eval split (post-filter): total={len(splits['eval'])}  "
            f"{_counts(splits['eval'])}",
            flush=True,
        )
    else:
        print(
            "[evaluate_autism] visual-token filter DISABLED — evaluating all "
            "videos (may OOM on 4K samples).",
            flush=True,
        )

    multi_dataset.get_all_samples = lambda *a, **k: splits

    # Forward to evaluate_multidataset.main with autism defaults layered in.
    sys.argv = [sys.argv[0]] + AUTISM_EVAL_DEFAULTS + extra_args

    # Patch evaluate_multidataset's scoring functions in-place so autism eval
    # uses set-equality for av_asd's multi-label answers (order-independent:
    # "A, B" == "B, A") while leaving evaluate_multidataset.py itself
    # untouched for the non-autism mix. multiple_choice (scrape_asd) keeps
    # the unchanged letter-extraction path.
    import evaluate_multidataset as em

    # No-pose mode: swap in a video-only dataset so __getitem__ skips
    # MediaPipe pose extraction entirely (the projector loaders already
    # return None for no-pose checkpoints, so the vanilla collate is used —
    # the only thing left to skip is the wasted pose-extraction work).
    if autism_args.no_pose:
        from train_avasd_no_pose import VideoOnlyDataset

        class _AutismVideoOnlyDataset(VideoOnlyDataset):
            """Disable VideoOnlyDataset's pre-scan filter so this stays
            drop-in compatible with how evaluate_multidataset constructs
            MultiVideoDataset (no scan_desc / max_visual_tokens kwargs).
            Oversized-video filtering already happens above in evaluate_autism
            when --filter_oversized_videos is on."""

            def __init__(self, samples, fps=2.0, max_frames=32, **_ignored):
                super().__init__(
                    samples, fps=fps, max_frames=max_frames,
                    skip_unreadable=False, scan_desc="(no-pose) scanning",
                    max_visual_tokens=None,
                )

        em.MultiVideoDataset = _AutismVideoOnlyDataset

        em.load_projector_from_hf    = lambda *a, **k: None
        em.load_projector_from_accel = lambda *a, **k: None

        print("[evaluate_autism] --no_pose: skipping MediaPipe pose extraction "
              "in dataset (video frames only) and disabling pose projector "
              "load (vanilla collate, plain model.generate).", flush=True)

    def _multilabel_set(text: str):
        return {em.normalize(p) for p in text.split(",") if em.normalize(p)}

    def _autism_is_correct(pred: str, gt: str, task_type: str) -> bool:
        if task_type == "multiple_choice":
            return em.extract_mc_letter(pred) == gt.strip().upper()
        # av_asd open_ended → multi-label set equality
        return _multilabel_set(pred) == _multilabel_set(gt)

    def _autism_is_correct_lenient(pred: str, gt: str, task_type: str) -> bool:
        if task_type == "multiple_choice":
            return em.extract_mc_letter(pred) == gt.strip().upper()
        # Lenient: prediction is correct if it covers all GT labels
        # (extra spurious labels allowed). Strict implies lenient.
        pred_set = _multilabel_set(pred)
        gt_set   = _multilabel_set(gt)
        if pred_set == gt_set:
            return True
        return bool(gt_set) and gt_set <= pred_set

    em.is_correct         = _autism_is_correct
    em.is_correct_lenient = _autism_is_correct_lenient

    from evaluate_multidataset import main as inner_main, parse_args as inner_parse
    inner_args = inner_parse()
    inner_main(inner_args)

    # Post-process the JSON evaluate_multidataset wrote so the autism-eval
    # output uses the same 4-metric scoring as train_avasd.py's val
    # (strict / f1 / hamming / balanced_acc). Re-scores from the per-sample
    # pred/gt already saved in the inner JSON — no re-generation needed.
    _emit_autism_4metric(inner_args.output_dir)


def _emit_autism_4metric(output_dir) -> None:
    """Locate the eval_*.json evaluate_multidataset just wrote, re-score every
    sample with train_avasd._score_one (strict + f1 + hamming + balanced_acc),
    pretty-print the per-dataset / per-task-type breakdown, and save a sibling
    ``*_autism4metric.json``."""
    import json
    from collections import defaultdict
    from pathlib import Path

    # Import the metric machinery from train_avasd so the eval scoring stays
    # in lock-step with training-time val.
    from train_avasd import _score_one, _METRIC_KEYS, _zero_metric_acc

    out_dir = Path(output_dir)
    candidates = [p for p in out_dir.glob("eval_*.json") if "autism4metric" not in p.name]
    if not candidates:
        print(f"[evaluate_autism] no eval_*.json found in {out_dir}; skipping 4-metric rescore.",
              flush=True)
        return
    src = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(src) as f:
        data = json.load(f)

    output = {
        "source_json": str(src),
        "timestamp":   data.get("timestamp"),
        "args":        data.get("args", {}),
        "metrics":     list(_METRIC_KEYS),
    }
    found_any = False

    for variant in ("finetuned", "pretrained"):
        samples = data.get(f"{variant}_samples", [])
        if not samples:
            continue
        found_any = True

        overall    = _zero_metric_acc()
        by_dataset = defaultdict(_zero_metric_acc)
        by_tt      = defaultdict(_zero_metric_acc)
        for s in samples:
            scores = _score_one(s["pred"], s["gt"], s["task_type"])
            for bucket in (overall, by_dataset[s["dataset"]], by_tt[s["task_type"]]):
                for m in _METRIC_KEYS:
                    bucket[m] += scores[m]
                bucket["total"] += 1

        def acc(b, k):
            return b[k] / max(1, b["total"])

        output[variant] = {
            "total": overall["total"],
            **{f"overall_{m}": acc(overall, m) for m in _METRIC_KEYS},
            "by_dataset": {
                ds: {**{m: acc(b, m) for m in _METRIC_KEYS}, "total": b["total"]}
                for ds, b in sorted(by_dataset.items())
            },
            "by_task_type": {
                tt: {**{m: acc(b, m) for m in _METRIC_KEYS}, "total": b["total"]}
                for tt, b in sorted(by_tt.items())
            },
        }

        print()
        print("=" * 78)
        print(f"  AUTISM 4-METRIC SCORING — {variant.upper()}  (re-scored from {src.name})")
        print("=" * 78)
        print(f"  total samples: {overall['total']}")
        print(f"  overall:        strict={acc(overall,'strict'):.4f}  "
              f"f1={acc(overall,'f1'):.4f}  "
              f"hamming={acc(overall,'hamming'):.4f}  "
              f"bal_acc={acc(overall,'balanced_acc'):.4f}")
        for ds in sorted(by_dataset):
            b = by_dataset[ds]
            print(f"    {ds:<12s} n={b['total']:>5d}  "
                  f"strict={acc(b,'strict'):.4f}  "
                  f"f1={acc(b,'f1'):.4f}  "
                  f"hamming={acc(b,'hamming'):.4f}  "
                  f"bal_acc={acc(b,'balanced_acc'):.4f}")
        for tt in sorted(by_tt):
            b = by_tt[tt]
            print(f"    {tt:<16s} n={b['total']:>5d}  "
                  f"strict={acc(b,'strict'):.4f}  "
                  f"f1={acc(b,'f1'):.4f}  "
                  f"hamming={acc(b,'hamming'):.4f}  "
                  f"bal_acc={acc(b,'balanced_acc'):.4f}")

    if not found_any:
        print(f"[evaluate_autism] {src.name} had no finetuned_samples / pretrained_samples; "
              f"skipping rescore.", flush=True)
        return

    dst = src.with_name(src.stem + "_autism4metric.json")
    with open(dst, "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"[evaluate_autism] 4-metric breakdown saved to: {dst}", flush=True)


if __name__ == "__main__":
    main()
