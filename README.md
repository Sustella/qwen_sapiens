# Qwen3.5 with Pose Feature Tokens

A customized version of **Qwen3.5-VL** (vision-language model) that augments the
standard video + text token stream with per-frame MediaPipe landmark embeddings.
The model is designed for behavior classification from video — specifically the
**AV-ASD** (Audio-Visual Autism Spectrum Dataset) task, where each clip is scored
for 9 autism-related behaviors.

---

## Architecture overview

```
Video frames ──► Qwen3.5 Vision Encoder ──► video token embeddings ─────┐
                                                                          │
Text prompt  ──► Token Embedder ──────────────────────────────────────────┤
                                                                          │
                                                     Qwen3.5 LLM (frozen) │
                                                     (32 hybrid layers)   │
                                                          │               │
                                                    hidden_states         │
                                                          │               ▼
Video path ──► OpenCV frame sampling                      │
           ──► MediaPipe (pose + face + hands)            │
           ──► flatten → (B, T, 1659)                     │
           ──► PoseFeatureProjector (2-layer MLP+GELU)    │
           ──► pose_embeds (B, T, hidden_size) ───────────┘
                                                          │
                                              cat([hidden_states, pose_embeds], dim=1)
                                                          │
                                                  lm_head → logits
                                                          │
                                              First generated token predicted
                                              from last pose embedding
```

### Pose feature projector

`PoseFeatureProjector` is a two-layer MLP:

```
Linear(1659, hidden_size) → GELU → Linear(hidden_size, hidden_size)
```

Default feature breakdown (1659-D total per frame):

| Modality   | Landmarks | Dims |
|------------|-----------|------|
| Pose body  | 33 × 3    | 99   |
| Face mesh  | 478 × 3   | 1434 |
| Left hand  | 21 × 3    | 63   |
| Right hand | 21 × 3    | 63   |

### How pose tokens influence generation

Pose embeddings are appended **after** the Qwen LLM hidden states (VideoNSA style):

```
hidden_states = cat([hidden_states, pose_embeds], dim=1)   # (B, seq+T, H)
logits = lm_head(hidden_states)
```

During **generation** (prefill step), the last position in the extended sequence is
the last pose token.  With `logits_to_keep=1` (standard for auto-regressive
generation) the LM head predicts the first response token directly from the final
pose embedding.  This means the model's initial prediction is conditioned on the
projected body/face/hand features.

During **training** pose positions receive labels `-100` and do not contribute to
the cross-entropy loss.  Gradients reach the projector through the generation-time
path (see Training section below).

---

## Files

```
qwen_model/
├── configuration_qwen3_5.py   # Qwen3_5Config — new fields: use_pose_features,
│                              #   pose_feature_dim, pose_sample_n
├── modeling_qwen3_5.py        # Full model (auto-generated copy, modified)
├── modular_qwen3_5.py         # Source-of-truth modular file, same changes
├── tokenization_qwen3_5.py    # Unchanged BPE tokenizer
└── utils/
    ├── pose_features.py       # MediaPipe extraction helpers
    └── extract_pose_features.py  # Standalone CLI for offline pre-extraction
```

---

## Dependencies

```bash
# Core
pip install transformers torch torchvision

# Pose extraction
pip install mediapipe opencv-python

# Training
pip install accelerate datasets peft trl

# Inference server
pip install vllm
```

MediaPipe model weights are downloaded automatically on first use.

---

## Configuration

Enable pose features by setting `use_pose_features=True` in the config:

```python
from configuration_qwen3_5 import Qwen3_5Config

config = Qwen3_5Config.from_pretrained("Qwen/Qwen3.5-9B")
config.use_pose_features = True
config.pose_feature_dim = 1659   # all four MediaPipe modalities
config.pose_sample_n = 16        # frames sampled per video at runtime
config.save_pretrained("my_qwen_pose/")
```

When `use_pose_features=True`, the model:

1. Creates `PoseFeatureProjector` automatically in `__init__`.
2. **Freezes the entire Qwen backbone** (Phase 1 — only the projector trains).
3. Expects a `video_paths` argument in `forward()` / `generate()`.

---

## Quick inference example

```python
import torch
from transformers import AutoProcessor
from modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "my_qwen_pose/", torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")

video_path = "/scratch/stellasu/clips_video_annotated/sample.mp4"

messages = [{
    "role": "user",
    "content": [
        {"type": "video_url", "video_url": {"url": video_path}},
        {"type": "text", "text": (
            "You are analyzing an AV-ASD video.\n"
            "For the behavior 'Upper Limb Stereotypies', indicate 1 if present and 0 if not.\n"
            "Output only a 0 or a 1."
        )},
    ],
}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        video_paths=[video_path],   # triggers ad-hoc pose extraction
        max_new_tokens=4,
    )

print(processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Training

### Why the projector can learn (training mechanics)

The Qwen backbone is frozen.  During training, examples are processed as:

```
prompt (video + text) → Qwen (frozen) → hidden_states
                                              │
                               append pose_embeds (projector)
                                              │
                         lm_head (frozen) → logits for ALL positions
```

With standard teacher-forcing and `logits_to_keep=0`, the loss is computed on
response token positions only (`labels != -100`).  Because pose tokens are
appended **after** the response in the sequence, they do **not** receive
gradients through the standard LM loss.

**To train the projector with gradients**, use one of:

#### Option A — generation-style training (recommended)

Process only the **prompt** (no response appended), then compute cross-entropy
on the first response token against the last pose token logit:

```python
# prompt_inputs: tokenized user message only (no assistant turn)
outputs = model(
    **prompt_inputs,
    video_paths=batch_video_paths,
    logits_to_keep=1,          # only compute last-position logit
)
# outputs.logits: (B, 1, vocab_size) — last pose token predicts first response token
first_token_ids = batch_first_response_token_ids  # e.g. id of "0" or "1"
loss = F.cross_entropy(outputs.logits[:, 0, :], first_token_ids)
loss.backward()
```

This path: `loss → lm_head(last pose token) → pose_projector` gives the
projector real gradients.

#### Option B — unfreeze lm_head too

Unfreeze the LM head so it fine-tunes alongside the projector.  Call
`model._freeze_backbone()` and then re-enable `lm_head`:

```python
for p in model.lm_head.parameters():
    p.requires_grad = True
```

#### Option C — Phase 2 (full fine-tuning with LoRA)

After Phase 1 converges, add LoRA adapters to the transformer layers:

```python
from peft import get_peft_model, LoraConfig

lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_cfg)
# pose_projector weights are still fully trainable (not wrapped by LoRA)
```

---

### Exact command to run

```bash
cd /home/ixzhu/qwen_model

python train.py \
    --model_name Qwen/Qwen3.5-VL-7B-Instruct \
    --train_data /home/ixzhu/orcd/pool/AV-ASD/AV-ASD/data_with_pose.jsonl \
    --val_data   /home/ixzhu/orcd/pool/AV-ASD/AV-ASD/val_data.jsonl \
    --output_dir /home/ixzhu/orcd/pool/qwen_pose \
    --pose_sample_n 16 \
    --n_frames 16 \
    --num_epochs 2 \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --log_steps 10
```

`--pose_feature_dim` is auto-detected from the first `.pt` file in the training set.
Checkpoints are saved per epoch under `output_dir/epoch-N/` and the best model (lowest val loss) is saved to `output_dir/best/`.

---

### Training data format

Create a JSONL file where each line is one training example:

```json
{
  "video_path": "/scratch/stellasu/clips_video_annotated/VIDEO_ID.mp4",
  "behavior": "Upper Limb Stereotypies",
  "label": 1
}
```

#### Convert AV-ASD CSV to training JSONL

```python
import json, pandas as pd

BEHAVIORS = [
    "Absence or Avoidance of Eye Contact", "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Non-Responsiveness to Verbal Interaction", "Non-Typical Language",
    "Object Lining-Up", "Self-Hitting or Self-Injurious Behavior",
    "Self-Spinning or Spinning Objects", "Upper Limb Stereotypies",
]
VIDEO_DIR = "/scratch/stellasu/clips_video_annotated"
CSV_PATH  = "/home/ixzhu/orcd/pool/AV-ASD/AV-ASD/dataset/csvs/train.csv"
OUT_PATH  = "train_pose.jsonl"

df = pd.read_csv(CSV_PATH)
with open(OUT_PATH, "w") as f:
    for _, row in df.iterrows():
        for behavior in BEHAVIORS:
            f.write(json.dumps({
                "video_path": f"{VIDEO_DIR}/{row['Video_ID']}.mp4",
                "behavior": behavior,
                "label": int(row[behavior]),
            }) + "\n")
```

#### Training loop (Option A, generation-style)

```python
import json, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

SYSTEM = "You are analyzing an AV-ASD video for autism-related behaviors."
PROMPT_TMPL = (
    "For the behavior '{behavior}', indicate 1 if present and 0 if not.\n"
    "Output only a 0 or a 1."
)

class AvasdDataset(Dataset):
    def __init__(self, jsonl_path):
        self.records = [json.loads(l) for l in open(jsonl_path)]
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

def collate(batch, processor, tokenizer):
    messages_list, video_paths, labels = [], [], []
    for rec in batch:
        messages_list.append([{
            "role": "system", "content": SYSTEM,
        }, {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": rec["video_path"]}},
                {"type": "text", "text": PROMPT_TMPL.format(behavior=rec["behavior"])},
            ],
        }])
        video_paths.append(rec["video_path"])
        labels.append(rec["label"])

    inputs = processor.apply_chat_template(
        messages_list, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    )
    label_ids = torch.tensor(
        [tokenizer.encode(str(l), add_special_tokens=False)[0] for l in labels]
    )
    return inputs, video_paths, label_ids

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "my_qwen_pose/", torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)

dataset = AvasdDataset("train_pose.jsonl")
loader  = DataLoader(dataset, batch_size=4, shuffle=True,
                     collate_fn=lambda b: collate(b, processor, processor.tokenizer))

model.train()
for epoch in range(10):
    for inputs, video_paths, label_ids in loader:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        label_ids = label_ids.to(model.device)

        outputs = model(
            **inputs,
            video_paths=video_paths,
            logits_to_keep=1,   # only last-position (last pose token) logit
        )
        # outputs.logits: (B, 1, vocab_size)
        loss = F.cross_entropy(outputs.logits[:, 0, :], label_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss={loss.item():.4f}")

model.save_pretrained("my_qwen_pose_trained/")
```

---

## Inference with vLLM

vLLM does not natively support `video_paths` in the request body, so the
recommended workflow is:

1. **Pre-extract** pose features to `.pt` files using the provided CLI:
   ```bash
   python utils/extract_pose_features.py \
       --input_dir /scratch/stellasu/clips_video_annotated \
       --output_dir /scratch/stellasu/pose_features \
       --sample_n 16 \
       --flatten
   ```

2. Load features in your client code and pass them as a tensor via a custom
   endpoint, **or** serve the model with a simple FastAPI wrapper instead of
   vLLM for production inference:

   ```python
   # fastapi_server.py (minimal example)
   from fastapi import FastAPI
   from pydantic import BaseModel
   import torch
   from modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
   from transformers import AutoProcessor

   app = FastAPI()
   model = Qwen3_5ForConditionalGeneration.from_pretrained(
       "my_qwen_pose_trained/", torch_dtype=torch.bfloat16, device_map="auto"
   )
   processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")

   class Request(BaseModel):
       video_path: str
       behavior: str

   @app.post("/predict")
   def predict(req: Request):
       messages = [{
           "role": "user", "content": [
               {"type": "video_url", "video_url": {"url": req.video_path}},
               {"type": "text", "text": f"For '{req.behavior}', output 0 or 1."},
           ],
       }]
       inputs = processor.apply_chat_template(
           messages, tokenize=True, add_generation_prompt=True,
           return_dict=True, return_tensors="pt",
       ).to(model.device)
       with torch.no_grad():
           ids = model.generate(**inputs, video_paths=[req.video_path], max_new_tokens=2)
       return {"prediction": processor.decode(ids[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip()}
   ```

   ```bash
   uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
   ```

3. For large-scale batched evaluation you can still use vLLM for the **base
   Qwen model** (no pose) and use the custom model only when pose conditioning
   is needed.

---

## Freezing strategy

### Phase 1 (default, `use_pose_features=True`)

Only `PoseFeatureProjector` trains (~8M–35M params depending on model size).
The LM backbone and LM head are frozen.  Fast to converge and avoids
catastrophic forgetting.  Use Option A (generation-style) training loss.

### Phase 2 (optional, after Phase 1 converges)

Unfreeze the LM head and/or add LoRA adapters to the transformer layers.
Train at a lower learning rate (1e-5 or less).

```python
# Unfreeze LM head only
for p in model.lm_head.parameters():
    p.requires_grad = True

# Or add LoRA to attention projections
from peft import get_peft_model, LoraConfig
model = get_peft_model(model, LoraConfig(r=16, target_modules=["q_proj","v_proj"]))
```

---

## Config reference

| Field               | Type  | Default | Description                                    |
|---------------------|-------|---------|------------------------------------------------|
| `use_pose_features` | bool  | `False` | Enable pose projector + backbone freeze        |
| `pose_feature_dim`  | int   | `1659`  | Input dimension of flattened MediaPipe vector  |
| `pose_sample_n`     | int   | `16`    | Frames sampled per video during forward pass   |
