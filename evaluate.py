#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained PoseFeatureProjector checkpoint on a test set.

Loads the projector saved by train.py (pose_projector.pt + projector_config.json)
alongside the frozen Qwen3.5 backbone and reports per-behavior and overall
accuracy, precision, recall, F1, and loss.

Usage
-----
  python evaluate.py \\
      --checkpoint /home/ixzhu/orcd/pool/qwen_pose/best \\
      --test_data  /home/ixzhu/orcd/pool/AV-ASD/AV-ASD/test_data.jsonl \\
      --n_frames 16
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------- #
#  Pose projector (must match train.py exactly)                                #
# --------------------------------------------------------------------------- #

class PoseFeatureProjector(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

def _linspace_indices(total: int, n: int) -> list[int]:
    if n >= total:
        return list(range(total))
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _load_pose_features(pose_path: str, target_n: int) -> torch.Tensor:
    raw = torch.load(pose_path, map_location="cpu", weights_only=False)

    if isinstance(raw, torch.Tensor):
        feat = raw.float()
    elif isinstance(raw, dict):
        parts = []
        for key in ("pose", "face", "left_hand", "right_hand"):
            if key in raw:
                v = raw[key]
                if isinstance(v, torch.Tensor):
                    parts.append(v.float())
        feat = torch.cat(parts, dim=-1) if parts else torch.zeros(1, 1659)
    else:
        feat = torch.zeros(target_n, 1659)

    if feat.shape[0] != target_n:
        feat = feat.T.unsqueeze(0)
        feat = F.interpolate(feat, size=target_n, mode="linear", align_corners=False)
        feat = feat.squeeze(0).T

    return feat


class AvasdPoseDataset(Dataset):
    def __init__(self, jsonl_path: str, n_frames: int = 16, pose_sample_n: int = 16):
        self.pose_sample_n = pose_sample_n
        self.examples: list[dict] = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                images_all = rec["images"]
                pose_path  = rec["pose_features_path"]
                messages   = rec["messages"]

                if not Path(pose_path).exists():
                    log.warning("Missing pose file, skipping: %s", pose_path)
                    continue

                indices = _linspace_indices(len(images_all), n_frames)
                images_sub = [images_all[i] for i in indices]

                for turn_idx in range(0, len(messages), 2):
                    user_msg = messages[turn_idx]["content"]
                    asst_msg = messages[turn_idx + 1]["content"] if turn_idx + 1 < len(messages) else None
                    if asst_msg is None:
                        continue

                    text  = user_msg.replace("<image>", "").strip()
                    label = asst_msg.strip()
                    if label not in ("0", "1"):
                        continue

                    # Match behavior name from question text
                    behavior = next((b for b in BEHAVIORS if b.lower() in text.lower()), "Unknown")

                    self.examples.append({
                        "images":    images_sub,
                        "pose_path": pose_path,
                        "text":      text,
                        "label":     label,
                        "behavior":  behavior,
                    })

        log.info("Loaded %d examples from %s", len(self.examples), jsonl_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        pil_images = []
        for p in ex["images"]:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception:
                pil_images.append(Image.new("RGB", (224, 224)))

        pose_feat = _load_pose_features(ex["pose_path"], self.pose_sample_n)
        return {
            "pil_images": pil_images,
            "pose_feat":  pose_feat,
            "text":       ex["text"],
            "label":      ex["label"],
            "behavior":   ex["behavior"],
        }


# --------------------------------------------------------------------------- #
#  Collate                                                                     #
# --------------------------------------------------------------------------- #

def _collate(batch: list[dict], processor, tokenizer, device: torch.device) -> dict:
    messages_batch = []
    for ex in batch:
        content = [{"type": "image"} for _ in ex["pil_images"]]
        content.append({"type": "text", "text": ex["text"]})
        messages_batch.append([{"role": "user", "content": content}])

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    all_images = [ex["pil_images"] for ex in batch]

    inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )

    pose_feat = torch.stack([ex["pose_feat"] for ex in batch], dim=0)
    label_ids = torch.tensor(
        [tokenizer.encode(ex["label"], add_special_tokens=False)[0] for ex in batch],
        dtype=torch.long,
    )
    labels_str = [ex["label"] for ex in batch]
    behaviors  = [ex["behavior"] for ex in batch]

    return {
        **{k: v.to(device) for k, v in inputs.items()},
        "pose_feat":  pose_feat.to(device),
        "label_ids":  label_ids.to(device),
        "labels_str": labels_str,
        "behaviors":  behaviors,
    }


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def parse_prediction(pred) -> int:
    """Parse a prediction value to 0 or 1. Returns None if unparsable."""
    try:
        val = int(pred)
        if val in (0, 1):
            return val
    except (ValueError, TypeError):
        pass

    pred_str = str(pred).strip()
    if pred_str and pred_str[0] == '0':
        return 0
    elif pred_str and pred_str[0] == '1':
        return 1

    return None


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics from ground truth and predictions."""
    tp = fp = fn = tn = 0
    unparsable = 0

    for true, pred in zip(y_true, y_pred):
        true = int(true)
        parsed_pred = parse_prediction(pred)

        if parsed_pred is None:
            unparsable += 1
            if true == 1:
                fn += 1
            else:
                fp += 1
            continue

        pred = parsed_pred
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'total': total, 'unparsable': unparsable,
        'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy,
        'precision': precision, 'recall': tpr, 'f1': f1,
        'tpr': tpr, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr,
    }


# --------------------------------------------------------------------------- #
#  Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def evaluate(args):
    ckpt_dir = Path(args.checkpoint)
    config_path = ckpt_dir / "projector_config.json"
    weights_path = ckpt_dir / "pose_projector.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"projector_config.json not found in {ckpt_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"pose_projector.pt not found in {ckpt_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    model_name   = args.model_name or cfg["model_name"]
    feat_dim     = cfg["pose_feature_dim"]
    hidden_size  = cfg["hidden_size"]
    pose_sample_n = args.pose_sample_n or cfg["pose_sample_n"]

    log.info("Loading backbone: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_device  = next(model.model.embed_tokens.parameters()).device
    lm_head_device = next(model.lm_head.parameters()).device
    log.info("embed_device: %s  lm_head_device: %s", embed_device, lm_head_device)

    log.info("Loading projector from %s", ckpt_dir)
    projector = PoseFeatureProjector(feat_dim, hidden_size)
    projector.load_state_dict(torch.load(weights_path, map_location="cpu"))
    projector = projector.to(device=lm_head_device, dtype=torch.bfloat16)
    projector.eval()

    ds = AvasdPoseDataset(args.test_data, n_frames=args.n_frames, pose_sample_n=pose_sample_n)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: _collate(b, processor, tokenizer, embed_device),
    )

    total_loss = 0.0
    behavior_data: dict[str, dict] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    all_y_true: list = []
    all_y_pred: list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pose_feat  = batch.pop("pose_feat").to(device=lm_head_device, dtype=torch.bfloat16)
            label_ids  = batch.pop("label_ids").to(lm_head_device)
            labels_str = batch.pop("labels_str")
            behaviors  = batch.pop("behaviors")

            backbone_out = model.model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
            )
            hidden = backbone_out.last_hidden_state

            pose_embeds = projector(pose_feat)
            extended    = torch.cat([hidden, pose_embeds], dim=1)
            logits      = model.lm_head(extended[:, -1:, :]).float()

            loss = F.cross_entropy(logits[:, 0, :], label_ids)
            total_loss += loss.item()

            preds = logits[:, 0, :].argmax(dim=-1)
            pred_strs = [tokenizer.decode([p]) for p in preds.tolist()]

            for pred_str, gt_str, behavior in zip(pred_strs, labels_str, behaviors):
                all_y_pred.append(pred_str.strip())
                all_y_true.append(gt_str.strip())
                behavior_data[behavior]["y_pred"].append(pred_str.strip())
                behavior_data[behavior]["y_true"].append(gt_str.strip())

    avg_loss = total_loss / max(1, len(loader))

    overall_metrics = compute_metrics(all_y_true, all_y_pred)
    behaviors = sorted(behavior_data.keys())

    # Header
    print("=" * 100)
    print("AV-ASD EVALUATION RESULTS ANALYSIS")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Total samples: {len(all_y_true)}")
    print("=" * 100)

    # Per-behavior metrics
    print("\n" + "-" * 100)
    print("PER-BEHAVIOR METRICS")
    print("-" * 100)

    for behavior in behaviors:
        data = behavior_data[behavior]
        metrics = compute_metrics(data["y_true"], data["y_pred"])

        print(f"\n{behavior}")
        print(f"  Samples: {metrics['total']} (Positive: {metrics['tp'] + metrics['fn']}, Negative: {metrics['tn'] + metrics['fp']}, Unparsable: {metrics['unparsable']})")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Score:          {metrics['f1']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall (TPR):      {metrics['recall']:.4f}")
        print(f"  Confusion Matrix:  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        print(f"  TP Rate (Sens.):   {metrics['tpr']:.4f}")
        print(f"  TN Rate (Spec.):   {metrics['tnr']:.4f}")
        print(f"  FP Rate:           {metrics['fpr']:.4f}")
        print(f"  FN Rate:           {metrics['fnr']:.4f}")

    # Overall metrics
    print("\n" + "-" * 100)
    print("OVERALL METRICS")
    print("-" * 100)

    print(f"\nTotal Samples: {overall_metrics['total']}")
    print(f"  Positive samples: {overall_metrics['tp'] + overall_metrics['fn']}")
    print(f"  Negative samples: {overall_metrics['tn'] + overall_metrics['fp']}")
    print(f"  Unparsable predictions: {overall_metrics['unparsable']} (counted as incorrect)")
    print(f"\nAccuracy:          {overall_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {overall_metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score:          {overall_metrics['f1']:.4f}")
    print(f"Precision:         {overall_metrics['precision']:.4f}")
    print(f"Recall (TPR):      {overall_metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (True Positive):  {overall_metrics['tp']}")
    print(f"  FP (False Positive): {overall_metrics['fp']}")
    print(f"  FN (False Negative): {overall_metrics['fn']}")
    print(f"  TN (True Negative):  {overall_metrics['tn']}")
    print(f"\nRates:")
    print(f"  TP Rate (Sensitivity): {overall_metrics['tpr']:.4f}")
    print(f"  TN Rate (Specificity): {overall_metrics['tnr']:.4f}")
    print(f"  FP Rate:               {overall_metrics['fpr']:.4f}")
    print(f"  FN Rate:               {overall_metrics['fnr']:.4f}")

    # Summary table
    print("\n" + "-" * 100)
    print("SUMMARY TABLE")
    print("-" * 100)
    header = f"{'Behavior':<45} {'Acc':>7} {'Bal.Acc':>8} {'F1':>7} {'TPR':>7} {'TNR':>7} {'FPR':>7} {'FNR':>7}"
    print(header)
    print("-" * len(header))

    for behavior in behaviors:
        data = behavior_data[behavior]
        m = compute_metrics(data["y_true"], data["y_pred"])
        print(f"{behavior:<45} {m['accuracy']:>7.4f} {m['balanced_accuracy']:>8.4f} {m['f1']:>7.4f} {m['tpr']:>7.4f} {m['tnr']:>7.4f} {m['fpr']:>7.4f} {m['fnr']:>7.4f}")

    print("-" * len(header))
    print(f"{'OVERALL':<45} {overall_metrics['accuracy']:>7.4f} {overall_metrics['balanced_accuracy']:>8.4f} {overall_metrics['f1']:>7.4f} {overall_metrics['tpr']:>7.4f} {overall_metrics['tnr']:>7.4f} {overall_metrics['fpr']:>7.4f} {overall_metrics['fnr']:>7.4f}")
    print("=" * 100)

    if args.output_json:
        results = {
            "checkpoint": str(ckpt_dir),
            "test_data": args.test_data,
            "loss": avg_loss,
            **overall_metrics,
            "per_behavior": {
                b: compute_metrics(behavior_data[b]["y_true"], behavior_data[b]["y_pred"])
                for b in BEHAVIORS
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results written to %s", args.output_json)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Qwen3.5 pose projector")
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint directory (contains pose_projector.pt + projector_config.json)",
    )
    p.add_argument(
        "--test_data", required=True,
        help="Path to test JSONL file",
    )
    p.add_argument(
        "--model_name", default=None,
        help="Override the backbone model name from the checkpoint config",
    )
    p.add_argument("--n_frames",     type=int, default=16)
    p.add_argument("--pose_sample_n", type=int, default=None,
                   help="Override pose_sample_n from checkpoint config")
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument(
        "--output_json", default=None,
        help="Optional path to write results as JSON",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
