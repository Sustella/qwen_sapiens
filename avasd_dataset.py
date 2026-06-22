"""
avasd_dataset.py — Data-loading utilities for the AV-ASD (Audio-Visual Autism
Spectrum Dataset) fine-tuning + evaluation pipeline.

Each CSV row describes one short video clip labelled with nine binary behavior
flags.  We expand each row into nine (video, pose, prompt, answer) tuples —
one per behavior — so the Qwen fine-tune gets per-behavior supervision.

Sample dict format (mirrors multi_dataset.py so the existing MultiVideoDataset
can consume it):
    video_path  : str
    pose_path   : str  (may or may not exist; skip_missing_pose filters)
    dataset     : 'avasd'
    task_type   : 'binary'  (prediction is "0" or "1")
    behavior    : str  (one of BEHAVIORS)
    question    : str  (prompt)
    answer      : str  (either "0" or "1")
    split       : 'train' | 'val' | 'eval'
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────

AVASD_ROOT      = Path("/home/ixzhu/orcd/pool/AV-ASD/AV-ASD/dataset")
AVASD_CSV_DIR   = AVASD_ROOT / "csvs"
AVASD_POSE_DIR  = AVASD_ROOT / "pose"
# Original (non-anonymized) video clips.  Pose features were extracted from
# these so the mesh and the frames line up.
AVASD_VIDEO_DIR = AVASD_ROOT / "clips_video"
# Anonymized (mesh-overlay-on-black) clips, used by evaluate_avasd.py.
AVASD_VIDEO_DIR_ANON = Path("/scratch/stellasu/clips_video_anonymized")

BEHAVIORS = [
    "Absence or Avoidance of Eye Contact",
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Non-Responsiveness to Verbal Interaction",
    "Non-Typical Language",
    "Object Lining-Up",
    "Self-Hitting or Self-Injurious Behavior",
    "Self-Spinning or Spinning Objects",
    "Upper Limb Stereotypies",
]

SPLIT_CSV = {
    "train": AVASD_CSV_DIR / "train.csv",
    "val":   AVASD_CSV_DIR / "val.csv",
    "eval":  AVASD_CSV_DIR / "test.csv",
}


# ── Prompt ───────────────────────────────────────────────────────────────────

def make_prompt(behavior: str, anonymized: bool = False) -> str:
    """Return the per-behavior prompt text.

    Matches evaluate_avasd.py so evaluation is apples-to-apples.
    The ``anonymized`` flag toggles the wording for mesh-overlay clips.
    """
    if anonymized:
        return (
            "You are a helpful assistant.\n"
            f"You are analyzing an AV-ASD (Audio-Visual Autism Spectrum "
            f"Dataset) video. For the behavior {behavior}, indicate 1 if the "
            f"behavior is present and 0 if not present.\n"
            "Note: Frames show pose, face, and hand meshes extracted from "
            "the original video and overlaid on black image for anonymity. "
            "Determine behaviors based off these meshes.\n"
            "Output only a 0 or a 1."
        )
    return (
        "You are a helpful assistant.\n"
        f"You are analyzing an AV-ASD (Audio-Visual Autism Spectrum Dataset) "
        f"video. For the behavior {behavior}, indicate 1 if the behavior is "
        f"present and 0 if not present.\n"
        "Output only a 0 or a 1."
    )


# ── Row → samples ────────────────────────────────────────────────────────────

def _row_to_samples(
    row: Dict[str, str],
    split: str,
    video_dir: Path,
    anonymized: bool,
    require_video: bool,
) -> List[dict]:
    video_id = row["Video_ID"]
    video_path = video_dir / f"{video_id}.mp4"
    pose_path = AVASD_POSE_DIR / f"{video_id}.pt"

    if require_video and not video_path.exists():
        return []

    samples = []
    for behavior in BEHAVIORS:
        gt = row.get(behavior, "").strip()
        if gt not in ("0", "1"):
            continue
        samples.append({
            "video_path": str(video_path),
            "pose_path":  str(pose_path),
            "dataset":    "avasd",
            "task_type":  "binary",
            "behavior":   behavior,
            "question":   make_prompt(behavior, anonymized=anonymized),
            "answer":     gt,
            "split":      split,
            "video_id":   video_id,
        })
    return samples


# ── Public API ───────────────────────────────────────────────────────────────

def get_avasd_samples(
    splits: Optional[List[str]] = None,
    anonymized: bool = False,
    require_video: bool = True,
) -> Dict[str, List[dict]]:
    """Load AV-ASD samples for the requested splits.

    Args
    ----
    splits: list of split names to load (default: all three).
    anonymized: use mesh-overlay clips + the matching prompt wording.
    require_video: drop rows whose .mp4 is missing.  Pose-missing rows are not
        dropped here (MultiVideoDataset's ``skip_missing_pose`` handles that).
    """
    if splits is None:
        splits = list(SPLIT_CSV.keys())

    video_dir = AVASD_VIDEO_DIR_ANON if anonymized else AVASD_VIDEO_DIR

    result: Dict[str, List[dict]] = {s: [] for s in splits}
    for split in splits:
        csv_path = SPLIT_CSV[split]
        if not csv_path.exists():
            print(f"  [AV-ASD] Missing CSV: {csv_path} — skipping {split}")
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                result[split].extend(_row_to_samples(
                    row, split, video_dir, anonymized, require_video,
                ))

    for split, samples in result.items():
        n_videos = len({s["video_id"] for s in samples})
        pos = sum(1 for s in samples if s["answer"] == "1")
        neg = sum(1 for s in samples if s["answer"] == "0")
        print(f"  [AV-ASD/{split}] {len(samples)} samples "
              f"({n_videos} videos, pos={pos}, neg={neg})")

    return result


if __name__ == "__main__":
    splits = get_avasd_samples()
    for split, samples in splits.items():
        if samples:
            print(f"\n--- {split} example ---")
            import json as _j
            print(_j.dumps(samples[0], indent=2))
