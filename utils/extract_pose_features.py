#!/usr/bin/env python3
"""
Standalone script for pre-extracting MediaPipe pose/face/hand features from videos.
This is used for preparing features for evaluation.

Usage:
    python extract_pose_features.py \
        --input_dir /path/to/videos \
        --output_dir /path/to/pose \
        --sample_n 128
"""

import argparse
import os
from glob import glob
from typing import List

import cv2
import torch
from tqdm import tqdm

from pose_features import (
    extract_features_for_frames,
    save_pose_features,
    compute_frame_indices,
    MEDIAPIPE_AVAILABLE,
)


def get_video_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """Get all video files in a directory."""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    video_files = []
    for ext in extensions:
        video_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
        video_files.extend(glob(os.path.join(input_dir, f'*{ext.upper()}')))

    return sorted(list(set(video_files)))


def extract_features_for_video(
    video_path: str,
    output_path: str,
    sample_n: int,
    feature_types: List[str] = None,
    flatten: bool = True,
) -> bool:
    """
    Extract features for a single video and save to file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get total frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            print(f"Video has no frames: {video_path}")
            return False

        # Compute frame indices to sample
        frame_indices = compute_frame_indices(total_frames, sample_n)

        # Extract features
        features = extract_features_for_frames(
            video_path,
            frame_indices,
            feature_types=feature_types
        )

        # Save features
        save_pose_features(
            features,
            output_path,
            flatten=flatten,
            feature_types=feature_types
        )

        return True

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe pose/face/hand features from videos.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing video files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save extracted features (.pt files).'
    )
    parser.add_argument(
        '--sample_n',
        type=int,
        default=128,
        help='Number of frames to sample from each video. Default: 128'
    )
    parser.add_argument(
        '--feature_types',
        nargs='+',
        default=['pose', 'face', 'left_hand', 'right_hand'],
        choices=['pose', 'face', 'left_hand', 'right_hand'],
        help='Feature types to extract. Default: all'
    )
    parser.add_argument(
        '--flatten',
        action='store_true',
        default=True,
        help='Flatten features into single tensor per video. Default: True'
    )
    parser.add_argument(
        '--no-flatten',
        dest='flatten',
        action='store_false',
        help='Save features as dictionary instead of flattened tensor.'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        help='Video file extensions to process.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files.'
    )

    args = parser.parse_args()

    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is not installed. Please install it with: pip install mediapipe")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get video files
    video_files = get_video_files(args.input_dir, args.extensions)

    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return 1

    print(f"Found {len(video_files)} video files")
    print(f"Extracting features with sample_n={args.sample_n}")
    print(f"Feature types: {args.feature_types}")
    print(f"Output directory: {args.output_dir}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for video_path in tqdm(video_files, desc="Extracting features"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(args.output_dir, f"{video_name}.pt")

        # Check if already exists
        if os.path.exists(output_path) and not args.overwrite:
            skip_count += 1
            continue

        success = extract_features_for_video(
            video_path=video_path,
            output_path=output_path,
            sample_n=args.sample_n,
            feature_types=args.feature_types,
            flatten=args.flatten,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDone!")
    print(f"  Processed: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
