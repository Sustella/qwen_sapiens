"""
Utilities for extracting and loading MediaPipe pose/face/hand features.
Compatible with MediaPipe 0.10.0+ (Tasks API).
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
MEDIAPIPE_AVAILABLE = True


# Feature dimensions
POSE_DIM = 33 * 3  # 33 landmarks * 3 coordinates (x, y, z)
FACE_DIM = 478 * 3  # 478 landmarks * 3 coordinates (updated for new API)
LEFT_HAND_DIM = 21 * 3  # 21 landmarks * 3 coordinates
RIGHT_HAND_DIM = 21 * 3  # 21 landmarks * 3 coordinates
TOTAL_DIM = POSE_DIM + FACE_DIM + LEFT_HAND_DIM + RIGHT_HAND_DIM

FEATURE_DIMS = {
    'pose': POSE_DIM,
    'face': FACE_DIM,
    'left_hand': LEFT_HAND_DIM,
    'right_hand': RIGHT_HAND_DIM,
}

# Default model paths - can be overridden via environment variables or function arguments
DEFAULT_MODEL_DIR = os.environ.get('MEDIAPIPE_MODEL_DIR', os.path.expanduser('~/.mediapipe/models'))
DEFAULT_POSE_MODEL = os.path.join(DEFAULT_MODEL_DIR, 'pose_landmarker_heavy.task')
DEFAULT_FACE_MODEL = os.path.join(DEFAULT_MODEL_DIR, 'face_landmarker.task')
DEFAULT_HAND_MODEL = os.path.join(DEFAULT_MODEL_DIR, 'hand_landmarker.task')


def _download_model_if_needed(model_path: str, model_url: str) -> str:
    """Download model file if it doesn't exist."""
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Downloading model to {model_path}...")
    import urllib.request
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

    return model_path


def get_model_paths(
    pose_model: Optional[str] = None,
    face_model: Optional[str] = None,
    hand_model: Optional[str] = None,
    auto_download: bool = True
) -> Dict[str, str]:
    """
    Get paths to MediaPipe model files, downloading if necessary.

    Args:
        pose_model: Path to pose landmarker model. If None, uses default.
        face_model: Path to face landmarker model. If None, uses default.
        hand_model: Path to hand landmarker model. If None, uses default.
        auto_download: If True, download models if they don't exist.

    Returns:
        Dictionary with 'pose', 'face', 'hand' keys mapping to model paths.
    """
    model_urls = {
        'pose': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
        'face': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
        'hand': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    }

    paths = {
        'pose': pose_model or DEFAULT_POSE_MODEL,
        'face': face_model or DEFAULT_FACE_MODEL,
        'hand': hand_model or DEFAULT_HAND_MODEL,
    }

    if auto_download:
        for key in paths:
            if not os.path.exists(paths[key]):
                _download_model_if_needed(paths[key], model_urls[key])

    return paths


class LandmarkerManager:
    """
    Manager class for MediaPipe landmarkers (pose, face, hands).
    Replaces the old Holistic model with separate task-based landmarkers.
    """

    def __init__(
        self,
        feature_types: List[str] = None,
        pose_model: Optional[str] = None,
        face_model: Optional[str] = None,
        hand_model: Optional[str] = None,
        running_mode: str = 'video',
        auto_download: bool = True
    ):
        """
        Initialize landmarkers for specified feature types.

        Args:
            feature_types: List of feature types to initialize.
                          Options: ['pose', 'face', 'left_hand', 'right_hand']
            pose_model: Path to pose landmarker model.
            face_model: Path to face landmarker model.
            hand_model: Path to hand landmarker model.
            running_mode: 'image' for single images, 'video' for video frames.
            auto_download: If True, download models if they don't exist.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Please install it with: pip install mediapipe")

        if feature_types is None:
            feature_types = ['pose', 'face', 'left_hand', 'right_hand']

        self.feature_types = feature_types
        self.running_mode = (
            vision.RunningMode.VIDEO if running_mode == 'video'
            else vision.RunningMode.IMAGE
        )

        # Get model paths
        model_paths = get_model_paths(pose_model, face_model, hand_model, auto_download)

        # Initialize landmarkers based on requested feature types
        self.pose_landmarker = None
        self.face_landmarker = None
        self.hand_landmarker = None

        if 'pose' in feature_types:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_paths['pose']),
                running_mode=self.running_mode,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        if 'face' in feature_types:
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_paths['face']),
                running_mode=self.running_mode,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        if 'left_hand' in feature_types or 'right_hand' in feature_types:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_paths['hand']),
                running_mode=self.running_mode,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        self._frame_timestamp_ms = 0

    def process(self, rgb_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a single RGB frame and extract features.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3)

        Returns:
            Dictionary mapping feature type to numpy array of flattened coordinates.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        features = {}

        if self.pose_landmarker is not None:
            if self.running_mode == vision.RunningMode.VIDEO:
                result = self.pose_landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
            else:
                result = self.pose_landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                features['pose'] = self._landmarks_to_array(result.pose_landmarks[0], 33)
            else:
                features['pose'] = np.zeros(POSE_DIM, dtype=np.float32)

        if self.face_landmarker is not None:
            if self.running_mode == vision.RunningMode.VIDEO:
                result = self.face_landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
            else:
                result = self.face_landmarker.detect(mp_image)

            if result.face_landmarks and len(result.face_landmarks) > 0:
                features['face'] = self._landmarks_to_array(result.face_landmarks[0], 478)
            else:
                features['face'] = np.zeros(FACE_DIM, dtype=np.float32)

        if self.hand_landmarker is not None:
            if self.running_mode == vision.RunningMode.VIDEO:
                result = self.hand_landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
            else:
                result = self.hand_landmarker.detect(mp_image)

            left_hand = np.zeros(LEFT_HAND_DIM, dtype=np.float32)
            right_hand = np.zeros(RIGHT_HAND_DIM, dtype=np.float32)

            if result.hand_landmarks and result.handedness:
                for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    # handedness[0].category_name is 'Left' or 'Right'
                    # Note: MediaPipe reports handedness from the camera's perspective,
                    # so 'Left' means the person's right hand and vice versa
                    hand_label = handedness[0].category_name
                    if hand_label == 'Right':  # Person's left hand
                        left_hand = self._landmarks_to_array(landmarks, 21)
                    else:  # Person's right hand
                        right_hand = self._landmarks_to_array(landmarks, 21)

            if 'left_hand' in self.feature_types:
                features['left_hand'] = left_hand
            if 'right_hand' in self.feature_types:
                features['right_hand'] = right_hand

        self._frame_timestamp_ms += 33  # Assume ~30fps for timestamp increment

        return features

    def _landmarks_to_array(self, landmarks, num_landmarks: int) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array."""
        coords = []
        for landmark in landmarks[:num_landmarks]:
            coords.extend([landmark.x, landmark.y, landmark.z])
        return np.array(coords, dtype=np.float32)

    def close(self):
        """Close all landmarkers and release resources."""
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _landmarks_to_array(landmarks, num_landmarks: int) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array."""
    if landmarks is None:
        return np.zeros(num_landmarks * 3, dtype=np.float32)

    coords = []
    for landmark in landmarks[:num_landmarks]:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.array(coords, dtype=np.float32)


def extract_features_for_frame(
    frame: np.ndarray,
    landmarker_manager: LandmarkerManager,
    feature_types: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract MediaPipe features for a single frame.

    Args:
        frame: BGR image as numpy array (H, W, 3)
        landmarker_manager: LandmarkerManager instance
        feature_types: List of feature types to extract.
                       Options: ['pose', 'face', 'left_hand', 'right_hand']
                       If None, extracts all features.

    Returns:
        Dictionary mapping feature type to numpy array of flattened coordinates.
    """
    if feature_types is None:
        feature_types = ['pose', 'face', 'left_hand', 'right_hand']

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    features = landmarker_manager.process(rgb_frame)

    # Filter to requested feature types
    return {ft: features.get(ft, np.zeros(FEATURE_DIMS[ft], dtype=np.float32))
            for ft in feature_types}


def extract_features_for_frames(
    video_path: str,
    frame_indices: List[int],
    feature_types: List[str] = None,
    pose_model: Optional[str] = None,
    face_model: Optional[str] = None,
    hand_model: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract MediaPipe features for specific frames in a video.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to extract features from.
        feature_types: List of feature types to extract.
                       Options: ['pose', 'face', 'left_hand', 'right_hand']
                       If None, extracts all features.
        pose_model: Path to pose landmarker model (optional).
        face_model: Path to face landmarker model (optional).
        hand_model: Path to hand landmarker model (optional).

    Returns:
        Dictionary mapping feature type to tensor of shape (num_frames, feature_dim).
    """
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError("MediaPipe is not installed. Please install it with: pip install mediapipe")

    if feature_types is None:
        feature_types = ['pose', 'face', 'left_hand', 'right_hand']

    # Initialize MediaPipe landmarkers
    landmarker = LandmarkerManager(
        feature_types=feature_types,
        pose_model=pose_model,
        face_model=face_model,
        hand_model=hand_model,
        running_mode='video'
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize feature storage
    all_features = {ft: [] for ft in feature_types}

    # Sort indices for sequential reading
    sorted_indices = sorted(enumerate(frame_indices), key=lambda x: x[1])

    current_frame_idx = 0

    for original_idx, target_frame in sorted_indices:
        # Seek to target frame
        if target_frame >= total_frames:
            # Use zeros for out-of-bounds frames
            for ft in feature_types:
                all_features[ft].append(np.zeros(FEATURE_DIMS[ft], dtype=np.float32))
            continue

        # Read frames until we reach the target
        while current_frame_idx <= target_frame:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_idx += 1

        if current_frame_idx <= target_frame:
            # Couldn't read frame, use zeros
            for ft in feature_types:
                all_features[ft].append(np.zeros(FEATURE_DIMS[ft], dtype=np.float32))
        else:
            # Extract features
            features = extract_features_for_frame(frame, landmarker, feature_types)
            for ft in feature_types:
                all_features[ft].append(features[ft])

    cap.release()
    landmarker.close()

    # Reorder features to match original frame_indices order
    reorder_map = [None] * len(frame_indices)
    for i, (original_idx, _) in enumerate(sorted_indices):
        reorder_map[original_idx] = i

    # Convert to tensors with correct order
    result = {}
    for ft in feature_types:
        ordered_features = [all_features[ft][reorder_map[i]] for i in range(len(frame_indices))]
        result[ft] = torch.from_numpy(np.stack(ordered_features, axis=0))

    return result


def flatten_features(
    features: Dict[str, torch.Tensor],
    feature_types: List[str] = None
) -> torch.Tensor:
    """
    Flatten feature dictionary into a single tensor.

    Args:
        features: Dictionary mapping feature type to tensor of shape (num_frames, feature_dim).
        feature_types: Order of features to concatenate. If None, uses default order.

    Returns:
        Tensor of shape (num_frames, total_feature_dim).
    """
    if feature_types is None:
        feature_types = ['pose', 'face', 'left_hand', 'right_hand']

    tensors = []
    for ft in feature_types:
        if ft in features:
            tensors.append(features[ft])

    if not tensors:
        raise ValueError("No features to flatten")

    return torch.cat(tensors, dim=-1)


def load_pose_features(
    features_path: str,
    feature_types: List[str] = None
) -> torch.Tensor:
    """
    Load pose features from a .pt file and return flattened tensor.

    Args:
        features_path: Path to the .pt file containing features.
        feature_types: List of feature types to include. If None, includes all.

    Returns:
        Tensor of shape (num_frames, feature_dim).
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features = torch.load(features_path)

    if feature_types is None:
        feature_types = ['pose', 'face', 'left_hand', 'right_hand']

    # If already flattened tensor, return as-is
    if isinstance(features, torch.Tensor):
        return features

    # Otherwise flatten the dictionary
    return flatten_features(features, feature_types)


def save_pose_features(
    features: Dict[str, torch.Tensor],
    output_path: str,
    flatten: bool = False,
    feature_types: List[str] = None
) -> None:
    """
    Save pose features to a .pt file.

    Args:
        features: Dictionary mapping feature type to tensor.
        output_path: Path to save the .pt file.
        flatten: If True, save as flattened tensor. Otherwise save as dict.
        feature_types: Feature types to include when flattening.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if flatten:
        tensor = flatten_features(features, feature_types)
        torch.save(tensor, output_path)
    else:
        torch.save(features, output_path)


def get_feature_dim(feature_types: List[str] = None) -> int:
    """
    Get total feature dimension for given feature types.

    Args:
        feature_types: List of feature types. If None, returns total dim for all types.

    Returns:
        Total feature dimension.
    """
    if feature_types is None:
        return TOTAL_DIM

    return sum(FEATURE_DIMS[ft] for ft in feature_types if ft in FEATURE_DIMS)


def compute_frame_indices(total_frames: int, sample_n: int) -> List[int]:
    """
    Compute evenly distributed frame indices for sampling.

    Args:
        total_frames: Total number of frames in the video.
        sample_n: Number of frames to sample.

    Returns:
        List of frame indices.
    """
    if sample_n is None or sample_n >= total_frames:
        return list(range(total_frames))

    if sample_n == 1:
        return [0]

    # Use linspace-like logic to get evenly spaced indices
    indices = [int(round(i * (total_frames - 1) / (sample_n - 1))) for i in range(sample_n)]
    return list(sorted(set(indices)))
