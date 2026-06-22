import mimetypes
import os
import time
from argparse import ArgumentParser
import torch
import multiprocessing as mp

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from tqdm import tqdm
import warnings
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from pathlib import Path
from typing import Dict, List, Optional

from mmdet.apis import inference_detector, init_detector
has_mmdet = True

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='mmengine')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.functional')
warnings.filterwarnings("ignore", category=UserWarning, module='json_tricks.encoders')

def safe_stack(feature_list, expected_shape):
    """Safely stack feature arrays, padding if necessary."""
    if not feature_list:
        return torch.zeros((0, *expected_shape), dtype=torch.float32)

    shapes = [arr.shape for arr in feature_list]
    if len(set(shapes)) == 1:
        return torch.tensor(np.array(feature_list), dtype=torch.float32)
    else:
        max_shape = [len(feature_list)]
        for dim in range(len(expected_shape)):
            max_shape.append(max(arr.shape[dim] if dim < len(arr.shape) else expected_shape[dim] 
                               for arr in feature_list))

        padded = np.zeros(max_shape, dtype=np.float32)
        for i, arr in enumerate(feature_list):
            if arr.ndim == len(expected_shape):
                slices = tuple(slice(None, s) for s in arr.shape)
                padded[i][slices] = arr
            else:
                padded[i][:arr.shape[0] if arr.ndim > 0 else 0] = arr.flatten()[:padded.shape[1]]

        return torch.tensor(padded, dtype=torch.float32)

def process_one_image(img,
                      detector,
                      pose_estimator):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    ## Defaults
    det_cat_id = 0
    bbox_thr = 0.3
    nms_thr = 0.3

    # if not isinstance(img, str):
    #     # Convert PIL Image to numpy array (RGB) then swap to BGR for OpenMMLab
    #     img = np.array(img)[:, :, ::-1]
    
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > bbox_thr)]
    bboxes = bboxes[nms(bboxes, nms_thr), :4]

    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    return data_samples.get('pred_instances', None)

def _extract_features_from_img_list(detector, pose_estimator, img_list) -> Dict[str, torch.Tensor]:
    bboxes = []
    bbox_scores = []
    keypts = [] 
    keypt_scores = []  
    
    for frame in img_list:
        pred_instances = process_one_image(frame, detector, pose_estimator)
        pred_dict = {key: pred_instances[key] for key in pred_instances.keys()} 
        
        if len(pred_dict["bbox_scores"]) >= 2:
            top2_indices = np.argsort(pred_dict["bbox_scores"])[-2:][::-1]
            for key in pred_dict.keys():
                pred_dict[key] = pred_dict[key][top2_indices]
        else: 
            miss_box = 2 - len(pred_dict["bbox_scores"])
            dummy_shapes = {
                "bboxes": (miss_box, 4),
                "bbox_scores": (miss_box,),
                "keypoints": (miss_box, 133, 2),
                # "keypoints_visible": (miss_box, 133),
                "keypoint_scores": (miss_box, 133)
            }
            for key, shape in dummy_shapes.items():
                data = pred_dict[key]
                dummy = np.zeros(shape, dtype=data.dtype)
                pred_dict[key] = np.concatenate([data, dummy], axis=0)
                
        bboxes.append(pred_dict["bboxes"])
        bbox_scores.append(pred_dict["bbox_scores"])
        keypts.append(pred_dict["keypoints"])
        keypt_scores.append(pred_dict["keypoint_scores"])
    
    features = {
        'bboxes': safe_stack(bboxes, (2, 4)),
        'bbox_scores': safe_stack(bbox_scores, (2, )),
        'keypoints': safe_stack(keypts, (2, 133, 2)),
        'keypoint_scores': safe_stack(keypt_scores, (2,133)),
    }
    return features
