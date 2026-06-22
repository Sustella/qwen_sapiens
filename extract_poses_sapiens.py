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

def _process_one(det, pose_est, s: dict, overwrite: bool) -> str:
    """Returns one of: 'ok', 'skip-existing', 'skip-bad', 'fail'."""
    out_path = Path(s["pose_path"])
    if not overwrite and out_path.exists():
        return "skip-existing"
    if not _is_video_readable(s["video_path"]):
        return "skip-bad"
        
    img_list = _video_frames(s["video_path"])
    feat = _extract_features_from_img_list(det, pose_est, img_list)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat, out_path)
    return "ok"

def _video_frames(video_path) -> List[Image.Image]:
    """
    Sample frames from a video at `fps` frames-per-second, capped at `max_frames`.
    Returns a variable-length list of PIL RGB images (at least 1).
    """

    cap = cv2.VideoCapture(video_path)
    img_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_list.append(frame)

    cap.release()

    # Return in order; substitute blank for any unreadable frame
    return img_list

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
    overwrite: bool,
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
        counts[_process_one(detector, pose_estimator, s, overwrite)] += 1

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
        "--overwrite", action="store_true",
        help="Re-extract even if the cache file already exists.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic dataset splits (default: 42).",
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
        print(f"Dataset: {dataset_name}")
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
                samples, args.overwrite
            )

    print("\nAll done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
