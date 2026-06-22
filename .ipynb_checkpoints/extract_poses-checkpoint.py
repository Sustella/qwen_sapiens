#!/usr/bin/env python3
# Suppress MediaPipe/GLOG C++ INFO and WARNING logs before any imports load the library.
import os
os.environ.setdefault("GLOG_minloglevel", "2")   # 0=INFO 1=WARNING 2=ERROR 3=FATAL

"""
extract_poses.py — Pre-extract MediaPipe pose/face/hand features for QVID,
Kinetics-400, and HMDB51.

Features are saved as .pt files (flattened tensor, shape: [sample_n, feat_dim])
to /orcd/compute/ppliang/001/poses/{QVID,kinetics,HMDB51}/...

The exact set of videos processed is determined by the same deterministic split
used during training (via multi_dataset.py), so you only pay the extraction cost
for videos that will actually be used.

Usage
-----
  # Single dataset
  python extract_poses.py --dataset qvid
  python extract_poses.py --dataset kinetics
  python extract_poses.py --dataset hmdb51

  # All at once
  python extract_poses.py --dataset all

  # Specific split only
  python extract_poses.py --dataset hmdb51 --split train

  # Re-extract (overwrite existing .pt files)
  python extract_poses.py --dataset qvid --overwrite
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2
import torch
from tqdm import tqdm

# Resolve pose_features.py from VideoNSA/utils
_UTILS_DIR = Path("/home/ixzhu/VideoNSA/utils")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from pose_features import (
    MEDIAPIPE_AVAILABLE,
    compute_frame_indices,
    extract_features_for_frames,
    save_pose_features,
)

from multi_dataset import get_all_samples, get_qvid_samples, get_kinetics_samples, get_hmdb51_samples

SAMPLE_N = 128          # frames to extract per video
FEATURE_TYPES = ["pose", "face", "left_hand", "right_hand"]


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_video(video_path: str, output_path: str, sample_n: int = 128) -> bool:
    """
    Extract MediaPipe features from a single video and save to output_path.
    Returns True on success, False on failure.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [WARN] Cannot open: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            print(f"  [WARN] Zero frames: {video_path}")
            return False

        frame_indices = compute_frame_indices(total_frames, sample_n)
        features = extract_features_for_frames(
            video_path, frame_indices, feature_types=FEATURE_TYPES
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_pose_features(features, output_path, flatten=True, feature_types=FEATURE_TYPES)
        return True

    except Exception as e:
        print(f"  [ERROR] {video_path}: {e}")
        return False


# ── Per-dataset drivers ───────────────────────────────────────────────────────

def process_samples(samples: List[dict], overwrite: bool = False, sample_n: int = 128) -> None:
    """Extract poses for a list of sample dicts (output of multi_dataset.py)."""
    to_process = []
    for s in samples:
        if not overwrite and Path(s["pose_path"]).exists():
            continue
        to_process.append(s)

    print(f"  Videos to process: {len(to_process)}  (skipping {len(samples) - len(to_process)} existing)")

    success, fail = 0, 0
    for s in tqdm(to_process, desc="Extracting"):
        ok = extract_video(s["video_path"], s["pose_path"], sample_n=sample_n)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"  Done — success: {success}  failed: {fail}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose features for multi-dataset training."
    )
    parser.add_argument(
        "--dataset",
        choices=["qvid", "kinetics", "hmdb51", "all"],
        default="all",
        help="Which dataset to process (default: all).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "eval", "all"],
        default="all",
        help="Which split to process (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if .pt file already exists.",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=SAMPLE_N,
        help=f"Frames to sample per video (default: {SAMPLE_N}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits (default: 42).",
    )
    args = parser.parse_args()

    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe not installed. Run: pip install mediapipe")
        return 1

    sample_n = args.sample_n

    # ── Load splits ───────────────────────────────────────────────────────────
    if args.dataset == "qvid":
        loaders = [("QVID", get_qvid_samples(args.seed))]
    elif args.dataset == "kinetics":
        loaders = [("Kinetics-400", get_kinetics_samples(args.seed))]
    elif args.dataset == "hmdb51":
        loaders = [("HMDB51", get_hmdb51_samples(args.seed))]
    else:  # all
        loaders = [
            ("QVID",         get_qvid_samples(args.seed)),
            ("Kinetics-400", get_kinetics_samples(args.seed)),
            ("HMDB51",       get_hmdb51_samples(args.seed)),
        ]

    # ── Extract ───────────────────────────────────────────────────────────────
    for dataset_name, splits in loaders:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        if args.split == "all":
            target_splits = ["train", "val", "eval"]
        else:
            target_splits = [args.split]

        for split_name in target_splits:
            samples = splits.get(split_name, [])
            if not samples:
                print(f"  [{split_name}] No samples found, skipping.")
                continue

            print(f"\n  [{split_name}] {len(samples)} videos")
            process_samples(samples, overwrite=args.overwrite, sample_n=sample_n)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
