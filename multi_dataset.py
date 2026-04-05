"""
multi_dataset.py — Shared data-loading utilities for QVID, Kinetics-400, and HMDB51.

Each sample dictionary contains:
    video_path  : str — absolute path to the video file
    pose_path   : str — where pose features are/will be saved (.pt)
    dataset     : str — 'qvid' | 'kinetics' | 'hmdb51'
    task_type   : str — 'open_ended' | 'multiple_choice'
    question    : str — question text (no <video> placeholders)
    answer      : str — target answer (letter for MC, short string for open-ended)
    split       : str — 'train' | 'val' | 'eval'
"""

import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# ── Dataset paths ────────────────────────────────────────────────────────────

QVID_DIR     = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/QVID")
KINETICS_DIR = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/kinetics-dataset/k400")
HMDB51_DIR   = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/HMDB51")
POSE_BASE    = Path("/orcd/compute/ppliang/001/poses")

# Kinetics subsample caps — keeps Kinetics roughly the same size as HMDB51 (~6766 total)
KINETICS_CAPS = {"train": 5600, "val": 700, "eval": 700}

SPLIT_SEED = 42  # fixed seed for all deterministic operations


# ── Deterministic 80/10/10 split ─────────────────────────────────────────────

def _assign_split(key: str, seed: int = SPLIT_SEED) -> str:
    """Return 'train', 'val', or 'eval' deterministically from a string key."""
    h = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16)
    r = (h % 1000) / 1000.0
    if r < 0.80:
        return "train"
    elif r < 0.90:
        return "val"
    else:
        return "eval"


# ── QVID ─────────────────────────────────────────────────────────────────────

def get_qvid_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load QVID labels.json and deterministically split 80/10/10.
    Returns {'train': [...], 'val': [...], 'eval': [...]}.
    """
    labels_path = QVID_DIR / "labels.json"
    with open(labels_path) as f:
        records = json.load(f)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}

    for rec in records:
        video_name = rec["video"]
        video_path = QVID_DIR / "videos" / video_name
        if not video_path.exists():
            continue

        stem = Path(video_name).stem
        pose_path = POSE_BASE / "QVID" / f"{stem}.pt"
        split = _assign_split(video_name, seed)

        splits[split].append({
            "video_path": str(video_path),
            "pose_path":  str(pose_path),
            "dataset":    "qvid",
            "task_type":  "open_ended",
            "question":   rec["question"] + " Answer concisely in a few words.",
            "answer":     rec["short_answer"],
            "split":      split,
        })

    return splits


# ── Kinetics-400 ─────────────────────────────────────────────────────────────

def get_kinetics_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load Kinetics-400, respecting existing train/val/test CSV splits.
    Each split is deterministically subsampled to KINETICS_CAPS.
    Returns {'train': [...], 'val': [...], 'eval': [...]}.
    """
    ann_dir = KINETICS_DIR / "annotations"

    # Map our split names to kinetics CSV / video dirs
    split_config = {
        "train": (ann_dir / "train.csv", KINETICS_DIR / "train"),
        "val":   (ann_dir / "val.csv",   KINETICS_DIR / "val"),
        "eval":  (ann_dir / "test.csv",  KINETICS_DIR / "test"),
    }

    result: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}

    for dest_split, (csv_path, video_dir) in split_config.items():
        cap = KINETICS_CAPS[dest_split]

        # Build set of existing filenames (one pass through directory)
        print(f"  [Kinetics] Indexing {video_dir} …")
        existing = {p.name for p in video_dir.iterdir() if p.suffix == ".mp4"}

        # Read annotations and filter to files that exist
        valid: List[Tuple[str, Path]] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    yt  = row["youtube_id"]
                    ts  = int(row["time_start"])
                    te  = int(row["time_end"])
                    lbl = row["label"]
                except (KeyError, ValueError):
                    continue
                fname = f"{yt}_{ts:06d}_{te:06d}.mp4"
                if fname in existing:
                    valid.append((lbl, video_dir / fname))

        # Deterministic subsample
        rng = random.Random(seed)
        if len(valid) > cap:
            valid = rng.sample(valid, cap)

        for label, video_path in valid:
            stem = video_path.stem
            pose_path = POSE_BASE / "kinetics" / dest_split / f"{stem}.pt"
            result[dest_split].append({
                "video_path": str(video_path),
                "pose_path":  str(pose_path),
                "dataset":    "kinetics",
                "task_type":  "open_ended",
                "question":   "What action is being performed in this video? Be concise.",
                "answer":     label.replace("_", " "),
                "split":      dest_split,
            })

    return result


# ── HMDB51 ───────────────────────────────────────────────────────────────────

def _build_hmdb51_file_index() -> Dict[str, Path]:
    """
    Scan all class directories and build a map: filename → full path.
    Skips .avi.avi duplicates and .tform.mat files.
    """
    index: Dict[str, Path] = {}
    for class_dir in HMDB51_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            name = f.name
            if name.endswith(".avi.avi") or name.endswith(".tform.mat"):
                continue
            if name.endswith(".avi"):
                index[name] = f
    return index


def get_hmdb51_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load HMDB51 labels.json (JSONL) and deterministically split 80/10/10.
    Returns {'train': [...], 'val': [...], 'eval': [...]}.
    """
    labels_path = HMDB51_DIR / "labels.json"

    print("  [HMDB51] Building file index …")
    file_index = _build_hmdb51_file_index()

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}

    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            videos_field = rec.get("videos", [])
            if not videos_field:
                continue

            video_entry = videos_field[0]
            video_filename = video_entry["video"]

            if video_filename not in file_index:
                continue
            video_path = file_index[video_filename]

            # Parse options and ground truth from the prompt
            user_content = rec["prompt"][1]["content"]
            option_matches = re.findall(r"([A-E])\. ([^\n]+)", user_content)
            options = {letter: label.strip() for letter, label in option_matches}
            ground_truth = rec["reward_model"]["ground_truth"]

            if not options or ground_truth not in options:
                continue

            # Build clean question with an instruction suffix
            base_q = " In the following video clip, what is the main human action being performed?"
            options_text = "\n".join(f"{l}. {v}" for l, v in options.items())
            question = (
                f"{base_q}\nOptions:\n{options_text}\n"
                "Output only the single letter (A, B, C, D, or E) corresponding to the correct answer."
            )

            stem = Path(video_filename).stem
            # Preserve class folder structure in pose path
            class_name = video_path.parent.name
            pose_path = POSE_BASE / "HMDB51" / class_name / f"{stem}.pt"

            split = _assign_split(video_filename, seed)
            splits[split].append({
                "video_path": str(video_path),
                "pose_path":  str(pose_path),
                "dataset":    "hmdb51",
                "task_type":  "multiple_choice",
                "question":   question,
                "answer":     ground_truth,
                "split":      split,
            })

    return splits


# ── Unified loader ────────────────────────────────────────────────────────────

def get_all_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load and merge all three datasets.
    Returns {'train': [...], 'val': [...], 'eval': [...]}.
    """
    print("[Dataset] Loading QVID …")
    qvid = get_qvid_samples(seed)

    print("[Dataset] Loading Kinetics-400 …")
    kinetics = get_kinetics_samples(seed)

    print("[Dataset] Loading HMDB51 …")
    hmdb51 = get_hmdb51_samples(seed)

    merged: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}
    for split in ("train", "val", "eval"):
        merged[split] = qvid[split] + kinetics[split] + hmdb51[split]

    for split, samples in merged.items():
        q  = sum(1 for s in samples if s["dataset"] == "qvid")
        k  = sum(1 for s in samples if s["dataset"] == "kinetics")
        h  = sum(1 for s in samples if s["dataset"] == "hmdb51")
        print(f"  [{split}] total={len(samples)}  qvid={q}  kinetics={k}  hmdb51={h}")

    return merged


if __name__ == "__main__":
    # Quick sanity check
    splits = get_all_samples()
    for split, samples in splits.items():
        if samples:
            print(f"\n--- {split} example ---")
            import json as _j
            print(_j.dumps(samples[0], indent=2))
