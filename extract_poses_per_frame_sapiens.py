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
import multiprocessing as mp
import sys
from pathlib import Path
from typing import List

import os
# Force PyTorch to bypass the strict weights_only restriction globally
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import torch
from tqdm import tqdm
import cv2
import torch
import numpy as np
from PIL import Image

from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.apis import init_model as init_pose_estimator
from utils.sapien_pose_features import _extract_features_from_img_list

# Reuse the EXACT same sampling + extraction code as train.py so the cache is
# guaranteed to match what online extraction would produce.
_REPO_DIR = Path(__file__).resolve().parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

# from train import (  # noqa: E402
    # _sample_frames,
    # _extract_pose_for_frames,
#     _per_frame_cache_path,
#     _is_video_readable,
# )
from multi_dataset import (  # noqa: E402
    get_qvid_samples,
    get_kinetics_samples,
    get_hmdb51_samples,
    # get_llava_video_samples,
    # get_sharegpt4video_samples,
    # get_av_asd_samples,
    # get_scrape_asd_samples,
)

try:
    import numpy as np
    import numpy.dtypes
    
    # 1. Add base types
    safe_list = [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    
    # 2. Dynamically find and add all hidden NumPy dtype classes automatically
    for item_name in dir(np.dtypes):
        item = getattr(np.dtypes, item_name)
        if isinstance(item, type):
            safe_list.append(item)
            
    torch.serialization.add_safe_globals(safe_list)
except Exception:
    pass

def _process_one(det, pose_est, s: dict, fps: float, max_frames: int, overwrite: bool) -> str:
    """Returns one of: 'ok', 'skip-existing', 'skip-bad', 'fail'."""
    out_path = _per_frame_cache_path(s, fps, max_frames)
    if not overwrite and out_path.exists():
        return "skip-existing"
    if not _is_video_readable(s["video_path"]):
        return "skip-bad"

    cap = cv2.VideoCapture(s["video_path"])
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    # Check if pose_path exists
    if Path(s["pose_path"]).exists():
        pose_feat = torch.load(s["pose_path"], map_location="cpu", weights_only=False)
        keypoints = pose_feat['keypoints']  # Shape: (N_frames, 2, 133, 2)
        scores = pose_feat['keypoint_scores']  # Shape: (N_frames, 2, 133)
        # Get Indices
       
        
        # total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = len(pose_feat["keypoints"])
        stride = max(1.0, video_fps / fps)
        indices: List[int] = []
        pos = 0.0
        while pos < total and len(indices) < max_frames:
            indices.append(min(round(pos), total - 1))
            pos += stride
        if not indices:
            indices = [0]
        indices = sorted(list(set(indices)))

        ## Normalize keypoints to 0 to 1
        x_coords = keypoints[..., 0]
        y_coords = keypoints[..., 1]
        x_norm = x_coords / width
        y_norm = y_coords / height
        normalized_keypoints = torch.stack([x_norm, y_norm], dim=-1)
        dummy_person = (scores.sum(dim=-1) == 0.0) 
        dummy_person = dummy_person.unsqueeze(-1).unsqueeze(-1)
        normalized_keypoints = torch.where(dummy_person, torch.tensor(-1.0), normalized_keypoints)
        
        # Flatten and Process
        pose_feat = torch.cat([normalized_keypoints, scores.unsqueeze(-1)], dim=-1)
        pose_feat = pose_feat[indices]
        pose_feat = pose_feat.flatten(start_dim=1)
        torch.save(pose_feat.contiguous().float().cpu(), out_path)
        return "ok"
        
    # try:
    pil_images = _sample_frames(s["video_path"], fps, max_frames)
    feat = _extract_features_from_img_list(det, pose_est, pil_images)

    keypoints = feat["keypoints"]
    scores = feat["keypoint_scores"]

    x_coords = keypoints[..., 0]
    y_coords = keypoints[..., 1]
    x_norm = x_coords / width
    y_norm = y_coords / height
    normalized_keypoints = torch.stack([x_norm, y_norm], dim=-1)
    dummy_person = (scores.sum(dim=-1) == 0.0) 
    dummy_person = dummy_person.unsqueeze(-1).unsqueeze(-1)
    normalized_keypoints = torch.where(dummy_person, torch.tensor(-1.0), normalized_keypoints)

    feat = torch.cat([normalized_keypoints, scores.unsqueeze(-1)], dim=-1)
    feat = feat.flatten(start_dim=1)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as CPU float32 tensor; train.py loads via torch.load(... ).float().
    torch.save(feat.contiguous().float().cpu(), out_path)
    return "ok"
    # except Exception as e:  # noqa: BLE001 — we want to log and continue
    #     print(f"  [ERROR] {s['video_path']}: {e}", flush=True)
    #     return "fail"

def _sample_frames(video_path: str, fps: float, max_frames: int) -> List[Image.Image]:
    """
    Sample frames from a video at `fps` frames-per-second, capped at `max_frames`.
    Returns a variable-length list of PIL RGB images (at least 1).
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return [Image.new("RGB", (224, 224))]

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame stride in video-frame units; at least 1
    stride = max(1.0, video_fps / fps)

    # Build target indices
    indices: List[int] = []
    pos = 0.0
    while pos < total and len(indices) < max_frames:
        indices.append(min(round(pos), total - 1))
        pos += stride

    if not indices:
        indices = [0]

    indices_set = set(indices)
    collected: dict = {}
    current = 0

    while current <= max(indices_set):
        ret, frame = cap.read()
        if not ret:
            break
        if current in indices_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            collected[current] = Image.fromarray(rgb)
        current += 1

    cap.release()

    # Return in order; substitute blank for any unreadable frame
    return [
        collected.get(i, Image.new("RGB", (224, 224)))
        for i in indices
    ]

def _per_frame_cache_path(s: dict, fps: float, max_frames: int) -> Path:
    """Cache path for offline-extracted per-frame pose features. Keyed by fps
    and max_frames so different sampling configs don't collide."""
    p = Path(s["pose_path"])
    return p.with_name(p.stem + f".perframe_fps{fps:.2f}_max{max_frames}.pt")


def _is_video_readable(video_path: str) -> bool:
    """Quick check: cv2 can open the file and it has at least one frame."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    finally:
        cap.release()

def _process_samples(
    name: str,
    samples: List[dict],
    fps: float,
    max_frames: int,
    overwrite: bool,
    workers: int = 1,
) -> None:
    counts = {"ok": 0, "skip-existing": 0, "skip-bad": 0, "fail": 0}

    # gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # Example: Change to your list of active GPUs, e.g., [0, 1, 2, 3]
    draw_heatmap = False # Set based on your args setup

    MODEL_NAME = 'sapiens_2b'
    DATASET = 'coco_wholebody'
    MODEL = f"{MODEL_NAME}-210e_{DATASET}-1024x768"
    
    det_cfg = '/home/stellasu/sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
    det_chpt = "/home/stellasu/scratch/stellasu/sapien_ckpts/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    
    pose_cfg = f"/home/stellasu/sapiens/pose/configs/sapiens_pose/{DATASET}/{MODEL}.py"
    pose_chpt = "/home/stellasu/scratch/stellasu/sapien_ckpts/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth"
    
    detector = init_detector(det_cfg, det_chpt, device='cuda:0') ## Device set as 0 temproarily
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_pose_estimator(
        pose_cfg,
        pose_chpt,
        device='cuda:0',
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))

    for s in tqdm(samples, desc=name, file=sys.stdout):
        counts[_process_one(detector, pose_estimator, s, fps, max_frames, overwrite)] += 1

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
        choices=[
            "qvid", "kinetics", "hmdb51",
            "llava_video", "sharegpt4video",
            "av_asd", "scrape_asd",
            "all",
        ],
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
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). MediaPipe is "
             "single-threaded, so set this to the number of CPU cores you "
             "want to dedicate (each worker spins up its own extractor).",
    )
    parser.add_argument(
        "--reverse", action='store_true',
        help="Reverse order of videos to process",
    )
    args = parser.parse_args()

    if args.dataset == "qvid":
        loaders = [("QVID", get_qvid_samples(args.seed))]
    elif args.dataset == "kinetics":
        loaders = [("Kinetics-400", get_kinetics_samples(args.seed))]
    elif args.dataset == "hmdb51":
        loaders = [("HMDB51", get_hmdb51_samples(args.seed))]
    elif args.dataset == "llava_video":
        loaders = [("LLaVA-Video", get_llava_video_samples(args.seed))]
    elif args.dataset == "sharegpt4video":
        loaders = [("ShareGPT4Video", get_sharegpt4video_samples(args.seed))]
    elif args.dataset == "av_asd":
        loaders = [("av-asd", get_av_asd_samples())]
    elif args.dataset == "scrape_asd":
        loaders = [("scrape_asd", get_scrape_asd_samples(args.seed))]
    else:
        loaders = [
            ("QVID",           get_qvid_samples(args.seed)),
            ("Kinetics-400",   get_kinetics_samples(args.seed)),
            ("HMDB51",         get_hmdb51_samples(args.seed)),
            ("LLaVA-Video",    get_llava_video_samples(args.seed)),
            ("ShareGPT4Video", get_sharegpt4video_samples(args.seed)),
            ("av-asd",         get_av_asd_samples()),
            ("scrape_asd",     get_scrape_asd_samples(args.seed)),
        ]

    target_splits = ["train", "val", "eval"] if args.split == "all" else [args.split]

    for dataset_name, splits in loaders:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  (fps={args.fps}, max_frames={args.max_frames})")
        print(f"{'='*60}")
        for split_name in target_splits:
            samples = splits.get(split_name, [])
            if args.reverse:
                samples.reverse()
            if not samples:
                print(f"  [{split_name}] no samples, skipping.")
                continue
            print(f"  [{split_name}] {len(samples)} videos")
            _process_samples(
                f"{dataset_name}/{split_name}",
                samples, args.fps, args.max_frames, args.overwrite,
                workers=args.workers,
            )

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
