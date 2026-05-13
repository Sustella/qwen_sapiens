"""
multi_dataset.py — Shared data-loading utilities for QVID, Kinetics-400, HMDB51,
ShareGPT4Video, LLaVA-Video, av-asd, and scrape_asd.

The default ``get_all_samples()`` mix is QVID + Kinetics-400 + HMDB51 +
ShareGPT4Video. LLaVA-Video and the two autism datasets are available as
standalone loaders but are not included in the default mix — opt in by
calling their loaders directly (or by patching ``get_all_samples`` from a
wrapper script, like ``train_autism.py`` does).

Each sample dictionary contains:
    video_path  : str — absolute path to the video file
    pose_path   : str — where pose features are/will be saved (.pt)
    dataset     : str — 'qvid' | 'kinetics' | 'hmdb51' | 'llava_video' | 'sharegpt4video'
    task_type   : str — 'open_ended' | 'multiple_choice'
    question    : str — question text (no <video> placeholders)
    answer      : str — target answer (letter for MC, short string for open-ended)
    split       : str — 'train' | 'val' | 'eval'
"""

import csv
import hashlib
import heapq
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# ── Dataset paths ────────────────────────────────────────────────────────────

QVID_DIR     = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/QVID")
KINETICS_DIR = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/kinetics-dataset/k400")
HMDB51_DIR   = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/HMDB51")
LLAVA_DIR    = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/LLaVA_Video")
GPT_DIR      = Path("/orcd/compute/ppliang/001/long_video_datasets/short_video/ShareGPT4Video")
AVASD_DIR    = Path("/orcd/compute/ppliang/001/jadali85/autism_datasets/av-asd")
SCRAPE_ASD_DIR = Path("/orcd/compute/ppliang/001/jadali85/autism_datasets/scrape_asd")
POSE_BASE    = Path("/orcd/compute/ppliang/001/poses")

# Subsample caps — keep each new dataset roughly the same size as HMDB51/Kinetics
# (~7000 total samples per dataset, split 80/10/10).
KINETICS_CAPS = {"train": 5600, "val": 700, "eval": 700}

# LLaVA-Video has 3 sub-label files (open-ended QA, multiple-choice, captioning).
# Cap each separately so the three task types are all represented, totalling ~7000.
LLAVA_CAPS = {
    "oe":  {"train": 3200, "val": 400, "eval": 400},   # open-ended QA
    "mc":  {"train": 1600, "val": 200, "eval": 200},   # multiple-choice
    "cap": {"train": 800,  "val": 100, "eval": 100},   # captioning (long answers)
}
GPT_CAPS  = {"train": 2800, "val": 300, "eval": 300}

SPLIT_SEED = 42  # fixed seed for all deterministic operations


# ── Deterministic hash helpers ───────────────────────────────────────────────

def _hash_uint64(*parts: str) -> int:
    """Deterministic 64-bit integer from string parts."""
    return int(hashlib.md5(":".join(parts).encode()).hexdigest()[:16], 16)


def _make_capped_collector(
    caps: Dict[str, int],
) -> Tuple[Callable[[str, str, dict], None], Callable[[], Dict[str, List[dict]]]]:
    """Build a streaming, deterministic top-k reservoir per split.

    Returns ``(add, finalize)`` where:
      * ``add(split, key, sample)`` keeps ``sample`` only if its hash of ``key``
        ranks among the smallest ``caps[split]`` seen so far for that split.
        Identical key → identical decision, regardless of stream length.
      * ``finalize()`` returns ``{split: [samples]}`` with at most ``cap`` each.

    Memory: O(sum(caps)). Independent of input stream size. The ordering of
    the same key is fixed across runs, so this is fully deterministic.
    """
    heaps: Dict[str, List[Tuple[int, int, dict]]] = {s: [] for s in caps}
    counters: Dict[str, int] = {s: 0 for s in caps}

    def add(split: str, key: str, sample: dict) -> None:
        cap = caps.get(split, 0)
        if cap <= 0:
            return
        h = _hash_uint64(key)
        heap = heaps[split]
        counters[split] += 1
        # Max-heap keyed on -h: heap[0] is the current largest h in the top-k.
        if len(heap) < cap:
            heapq.heappush(heap, (-h, counters[split], sample))
        elif -h > heap[0][0]:  # i.e. this h is smaller than the current max
            heapq.heapreplace(heap, (-h, counters[split], sample))

    def finalize() -> Dict[str, List[dict]]:
        return {s: [item[2] for item in heaps[s]] for s in heaps}

    return add, finalize


# ── Prompt cleaning ──────────────────────────────────────────────────────────

_VIDEO_TOKEN_RE = re.compile(r"<video>\s*\n*")
_IMAGE_TOKEN_RE = re.compile(r"<image>\s*\n*")
_MC_INSTR_TAIL_RE = re.compile(
    r"\n*Please (?:provide|respond|answer|select).*$",
    flags=re.IGNORECASE | re.DOTALL,
)

def _clean_user_content(content: str) -> str:
    """Strip <video>/<image> placeholders that are inserted by the processor."""
    text = _VIDEO_TOKEN_RE.sub("", content)
    text = _IMAGE_TOKEN_RE.sub("", text)
    return text.strip()


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


# ── LLaVA-Video ──────────────────────────────────────────────────────────────

def _llava_pose_path(video_rel: str) -> Path:
    """POSE_BASE/LLaVA_Video/<same hierarchy as video_rel, but .pt>."""
    rel = Path(video_rel)
    return POSE_BASE / "LLaVA_Video" / rel.parent / f"{rel.stem}.pt"


def _stream_llava_subfile(
    label_path: Path,
    task_type: str,
    is_mc: bool,
    add: Callable[[str, str, dict], None],
    seed: int,
) -> None:
    """Stream a single LLaVA-Video JSONL into the capped collector."""
    if not label_path.exists():
        print(f"  [LLaVA] missing {label_path.name} — skipped")
        return

    print(f"  [LLaVA] streaming {label_path.name} …")
    sub_tag = label_path.stem  # e.g. "labels_oe"
    n_seen = 0
    with open(label_path) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            videos_field = rec.get("videos") or []
            if not videos_field:
                continue
            video_rel = videos_field[0].get("video", "")
            if not video_rel:
                continue

            try:
                user_content = rec["prompt"][1]["content"]
                gt = rec["reward_model"]["ground_truth"]
            except (KeyError, IndexError, TypeError):
                continue
            if not gt:
                continue

            question = _clean_user_content(user_content)
            if is_mc:
                # Replace whatever trailing "Please respond..." instruction is
                # there with the project-standard MC suffix used by HMDB51.
                question = _MC_INSTR_TAIL_RE.sub("", question).strip()
                question = (
                    f"{question}\n"
                    "Output only the single letter (A, B, C, D, or E) "
                    "corresponding to the correct answer."
                )
                gt = gt.strip().upper()[:1]
                if gt not in "ABCDE":
                    continue

            # Each entry is unique (video, sub-file, line). Hash on the tuple
            # so that the same row always lands in the same split/keep-set.
            key = f"llava:{sub_tag}:{video_rel}:{line_idx}"
            split = _assign_split(key, seed)

            sample = {
                "video_path": str(LLAVA_DIR / video_rel),
                "pose_path":  str(_llava_pose_path(video_rel)),
                "dataset":    "llava_video",
                "task_type":  task_type,
                "question":   question,
                "answer":     gt,
                "split":      split,
            }
            add(split, key, sample)
            n_seen += 1
    print(f"  [LLaVA]   {label_path.name}: streamed {n_seen} rows")


def get_llava_video_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load LLaVA-Video by streaming its 3 JSONL label files (oe / mc / cap).

    Each sub-file is capped independently so all three task styles survive.
    Uses a deterministic top-k reservoir keyed on row hash — never materialises
    the full 4.4M-line open-ended file in memory.
    """
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}

    sub_files = [
        ("labels_oe.json",  "open_ended",      False, LLAVA_CAPS["oe"]),
        ("labels_mc.json",  "multiple_choice", True,  LLAVA_CAPS["mc"]),
        ("labels_cap.json", "open_ended",      False, LLAVA_CAPS["cap"]),
    ]

    for fname, task_type, is_mc, caps in sub_files:
        add, finalize = _make_capped_collector(caps)
        _stream_llava_subfile(LLAVA_DIR / fname, task_type, is_mc, add, seed)
        for split, items in finalize().items():
            splits[split].extend(items)

    return splits


# ── ShareGPT4Video ───────────────────────────────────────────────────────────

def get_sharegpt4video_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load ShareGPT4Video labels.json (JSONL captioning data). Video files live
    under ``zip_folder/``, so we prefix the relative paths from the labels.
    """
    label_path = GPT_DIR / "labels.json"
    if not label_path.exists():
        print(f"  [ShareGPT4Video] missing {label_path} — skipped")
        return {"train": [], "val": [], "eval": []}

    print(f"  [ShareGPT4Video] streaming {label_path.name} …")
    add, finalize = _make_capped_collector(GPT_CAPS)
    n_seen = 0
    with open(label_path) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            videos_field = rec.get("videos") or []
            if not videos_field:
                continue
            video_rel = videos_field[0].get("video", "")
            if not video_rel:
                continue

            try:
                user_content = rec["prompt"][1]["content"]
                gt = rec["reward_model"]["ground_truth"]
            except (KeyError, IndexError, TypeError):
                continue
            if not gt:
                continue

            question = _clean_user_content(user_content)
            rel = Path(video_rel)
            video_full = GPT_DIR / "zip_folder" / rel
            pose_full = POSE_BASE / "ShareGPT4Video" / rel.parent / f"{rel.stem}.pt"

            key = f"sharegpt4video:{video_rel}:{line_idx}"
            split = _assign_split(key, seed)

            add(split, key, {
                "video_path": str(video_full),
                "pose_path":  str(pose_full),
                "dataset":    "sharegpt4video",
                "task_type":  "open_ended",
                "question":   question,
                "answer":     gt,
                "split":      split,
            })
            n_seen += 1

    print(f"  [ShareGPT4Video] streamed {n_seen} rows")
    return finalize()


# ── av-asd (autism behavior multi-label) ─────────────────────────────────────

# Strip the leading `<audio>\n` and `<video>\n` placeholders from the av-asd
# prompts. The training collate function re-inserts video tokens itself; the
# model has no audio encoder, so the audio placeholder must be removed.
_AVASD_AUDIO_RE = re.compile(r"<audio>\s*\n*")


def get_av_asd_samples(
    variant: str = "multilabel",
    seed: int = SPLIT_SEED,
) -> Dict[str, List[dict]]:
    """
    Load the av-asd autism behaviour dataset.

    The dataset ships with pre-built train/val/test splits as JSONL files, so we
    respect those rather than re-splitting deterministically. Test maps to
    `eval` to match the rest of the pipeline.

    Args:
        variant: ``"multilabel"`` reads ``av_asd_promptsmultilabel_{split}.jsonl``
            where the answer is a comma-separated list of behaviour names
            (~30 chars). Other variants are not currently supported.
        seed: unused (the splits are fixed by the dataset), kept for signature
            symmetry with the other loaders.
    """
    if variant != "multilabel":
        raise ValueError(f"av-asd variant {variant!r} not supported")

    split_files = {
        "train": "av_asd_promptsmultilabel_train.jsonl",
        "val":   "av_asd_promptsmultilabel_val.jsonl",
        "eval":  "av_asd_promptsmultilabel_test.jsonl",
    }

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}

    for split_name, fname in split_files.items():
        path = AVASD_DIR / fname
        if not path.exists():
            print(f"  [av-asd] missing {path.name} — skipping {split_name}")
            continue

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                videos_field = rec.get("videos") or []
                if not videos_field:
                    continue
                video_rel = videos_field[0]
                if not video_rel:
                    continue

                problem = rec.get("problem", "")
                answer  = rec.get("answer", "")
                if not problem or not answer:
                    continue

                # Strip <audio> (no audio encoder) and <video> (the collate
                # function adds video tokens itself). Keep the rest of the
                # prompt — including the "Based on the audio and video"
                # wording — intact, per the original dataset.
                question = _AVASD_AUDIO_RE.sub("", problem)
                question = _VIDEO_TOKEN_RE.sub("", question).strip()

                rel = Path(video_rel)
                stem = rel.stem
                splits[split_name].append({
                    "video_path": str(AVASD_DIR / rel),
                    "pose_path":  str(POSE_BASE / "av_asd" / f"{stem}.pt"),
                    "dataset":    "av_asd",
                    "task_type":  "open_ended",
                    "question":   question,
                    "answer":     answer,
                    "split":      split_name,
                })

    return splits


# ── scrape_asd (FSU ASD Video Glossary, binary) ──────────────────────────────

_SCRAPE_ASD_DIAGNOSTIC_TYPES = ("Typical", "Red Flags for ASD")


def get_scrape_asd_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load the scrape_asd dataset (FSU ASD Video Glossary diagnostic clips).

    There are 238 clips total in ``dataset.json``; we keep only the 134
    "Typical" / "Red Flags for ASD" diagnostic clips (the other 104 are
    "Treatments" with no diagnostic label). No built-in train/val/test
    split exists, so we deterministically split 80/10/10 by ``bcID`` hash,
    mirroring QVID/HMDB51.

    Formulated as a 2-option ``multiple_choice`` task — answer is the letter
    A (Typical) or B (Red Flags for ASD), matching how HMDB51 is set up so
    the existing letter-extraction evaluator works without changes.
    """
    catalog_path = SCRAPE_ASD_DIR / "dataset.json"
    if not catalog_path.exists():
        print(f"  [scrape_asd] missing {catalog_path} — skipped")
        return {"train": [], "val": [], "eval": []}

    with open(catalog_path) as f:
        catalog = json.load(f)

    label_to_letter = {"Typical": "A", "Red Flags for ASD": "B"}

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}

    for rec in catalog:
        typ = rec.get("type")
        if typ not in _SCRAPE_ASD_DIAGNOSTIC_TYPES:
            continue
        video_rel = rec.get("video_file") or ""
        if not video_rel:
            continue
        video_path = SCRAPE_ASD_DIR / video_rel
        if not video_path.exists():
            continue

        bc_id = rec["bcID"]
        section = rec.get("section_text") or ""
        sub = rec.get("subsection_text")
        section_def = (rec.get("section_definition") or "").strip()
        location = section + (f" — {sub}" if sub else "")

        # Prompt format mirrors the existing scrape_asd/build_sft.py framing
        # so the model sees a familiar taxonomy-anchored question.
        parts = [
            f"This clip is from the ASD Video Glossary in the section '{location}'.",
        ]
        if section_def:
            parts.append(f"Definition: {section_def}")
        parts.extend([
            "",
            "Which of the following best describes what is shown?",
            "A. Typical",
            "B. Red Flags for ASD",
            "Output only the single letter (A, B, C, D, or E) corresponding to the correct answer.",
        ])
        question = "\n".join(parts)
        answer = label_to_letter[typ]

        split = _assign_split(f"scrape_asd:{bc_id}", seed)
        stem = Path(video_rel).stem

        splits[split].append({
            "video_path": str(video_path),
            "pose_path":  str(POSE_BASE / "scrape_asd" / f"{stem}.pt"),
            "dataset":    "scrape_asd",
            "task_type":  "multiple_choice",
            "question":   question,
            "answer":     answer,
            "split":      split,
        })

    return splits


# ── Unified loader ────────────────────────────────────────────────────────────

def get_all_samples(seed: int = SPLIT_SEED) -> Dict[str, List[dict]]:
    """
    Load and merge the default multi-dataset training mix:
    QVID + Kinetics-400 + HMDB51 + ShareGPT4Video.

    LLaVA-Video is intentionally NOT included by default. Streaming its
    4.4M-line ``labels_oe.json`` takes ~17 minutes per process, which under
    ``accelerate launch --num_processes N`` would be paid by every rank on
    every launch and dominates wall time before training even starts. The
    ``get_llava_video_samples`` loader and its pose-extraction wiring in
    ``extract_poses_per_frame.py`` still exist; opt in explicitly if you want
    LLaVA-Video in a particular run.

    Returns {'train': [...], 'val': [...], 'eval': [...]}.
    """
    print("[Dataset] Loading QVID …")
    qvid = get_qvid_samples(seed)

    print("[Dataset] Loading Kinetics-400 …")
    kinetics = get_kinetics_samples(seed)

    print("[Dataset] Loading HMDB51 …")
    hmdb51 = get_hmdb51_samples(seed)

    print("[Dataset] Loading ShareGPT4Video …")
    sharegpt = get_sharegpt4video_samples(seed)

    merged: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}
    for split in ("train", "val", "eval"):
        merged[split] = (
            qvid[split]
            + kinetics[split]
            + hmdb51[split]
            + sharegpt[split]
        )

    for split, samples in merged.items():
        counts = {
            "qvid":          0,
            "kinetics":      0,
            "hmdb51":        0,
            "sharegpt4video":0,
        }
        for s in samples:
            counts[s["dataset"]] = counts.get(s["dataset"], 0) + 1
        counts_str = "  ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  [{split}] total={len(samples)}  {counts_str}")

    return merged


if __name__ == "__main__":
    # Quick sanity check
    splits = get_all_samples()
    for split, samples in splits.items():
        if samples:
            print(f"\n--- {split} example ---")
            import json as _j
            print(_j.dumps(samples[0], indent=2))
