#!/usr/bin/env python3
# Suppress MediaPipe / TFLite stderr spam before any imports.
import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

"""
extract_poses_per_frame.py — Offline pre-extraction of per-frame pose features
that match what train.py extracts ONLINE.

The on-disk artifact is shape ``(N_frames, 1659)`` per video, where N_frames
is the number of frames the model would sample given ``--fps`` and
``--max_frames`` (same logic as ``_sample_frames`` in train.py). Frames are
processed in temporal order with a fresh ``LandmarkerManager(running_mode=
'video')`` per video, so the resulting features are byte-identical to what
the online extractor in train.py would produce.

This is purely a speed optimization. ``MultiVideoDataset.__getitem__`` will
prefer the cache file if present, and fall back to online extraction otherwise.

Usage
-----
  # All datasets, all splits, fps=1 max_frames=8 (training defaults)
  python extract_poses_per_frame.py --fps 1.0 --max_frames 8

  # One dataset / split
  python extract_poses_per_frame.py --dataset qvid --split train --fps 1.0 --max_frames 8

  # Re-extract (overwrite existing cache files)
  python extract_poses_per_frame.py --overwrite --fps 1.0 --max_frames 8
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# Reuse the EXACT same sampling + extraction code as train.py so the cache is
# guaranteed to match what online extraction would produce.
_REPO_DIR = Path(__file__).resolve().parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from train import (  # noqa: E402
    _sample_frames,
    _extract_pose_for_frames,
    _per_frame_cache_path,
    _is_video_readable,
)
from multi_dataset import (  # noqa: E402
    get_qvid_samples,
    get_kinetics_samples,
    get_hmdb51_samples,
)


def _process_one(s: dict, fps: float, max_frames: int, overwrite: bool) -> str:
    """Returns one of: 'ok', 'skip-existing', 'skip-bad', 'fail'."""
    out_path = _per_frame_cache_path(s, fps, max_frames)
    if not overwrite and out_path.exists():
        return "skip-existing"
    if not _is_video_readable(s["video_path"]):
        return "skip-bad"
    try:
        pil_images = _sample_frames(s["video_path"], fps, max_frames)
        feat = _extract_pose_for_frames(pil_images)  # (N, 1659)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as CPU float32 tensor; train.py loads via torch.load(... ).float().
        torch.save(feat.contiguous().float().cpu(), out_path)
        return "ok"
    except Exception as e:  # noqa: BLE001 — we want to log and continue
        print(f"  [ERROR] {s['video_path']}: {e}", flush=True)
        return "fail"


def _process_samples(
    name: str, samples: List[dict], fps: float, max_frames: int, overwrite: bool,
) -> None:
    counts = {"ok": 0, "skip-existing": 0, "skip-bad": 0, "fail": 0}
    for s in tqdm(samples, desc=name, file=sys.stdout):
        counts[_process_one(s, fps, max_frames, overwrite)] += 1
    print(
        f"  {name}: ok={counts['ok']} skip-existing={counts['skip-existing']} "
        f"skip-bad={counts['skip-bad']} fail={counts['fail']}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-extract per-frame MediaPipe pose features matching "
                    "train.py's online extractor."
    )
    parser.add_argument(
        "--dataset",
        choices=["qvid", "kinetics", "hmdb51", "all"],
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "eval", "all"],
        default="all",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Target sampling rate (frames/sec). Must match training.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=16,
        help="Cap on sampled frames per video. Must match training.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if the cache file already exists.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic dataset splits (default: 42).",
    )
    args = parser.parse_args()

    if args.dataset == "qvid":
        loaders = [("QVID", get_qvid_samples(args.seed))]
    elif args.dataset == "kinetics":
        loaders = [("Kinetics-400", get_kinetics_samples(args.seed))]
    elif args.dataset == "hmdb51":
        loaders = [("HMDB51", get_hmdb51_samples(args.seed))]
    else:
        loaders = [
            ("QVID",         get_qvid_samples(args.seed)),
            ("Kinetics-400", get_kinetics_samples(args.seed)),
            ("HMDB51",       get_hmdb51_samples(args.seed)),
        ]

    target_splits = ["train", "val", "eval"] if args.split == "all" else [args.split]

    for dataset_name, splits in loaders:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  (fps={args.fps}, max_frames={args.max_frames})")
        print(f"{'='*60}")
        for split_name in target_splits:
            samples = splits.get(split_name, [])
            if not samples:
                print(f"  [{split_name}] no samples, skipping.")
                continue
            print(f"  [{split_name}] {len(samples)} videos")
            _process_samples(
                f"{dataset_name}/{split_name}",
                samples, args.fps, args.max_frames, args.overwrite,
            )

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
