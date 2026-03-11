# 🦷 SmileGuard — Full Project Handoff Brief
> **For:** Claude (VS Code / GitHub Copilot extension)
> **From:** Claude (claude.ai conversation)
> **Purpose:** Complete context transfer so you can continue this project without gaps
> **Date:** March 11, 2026

---

## 🧠 TL;DR — What You Need To Know Immediately

The user (`nangi` based on file paths) is building a **YOLOv8 dental disease detection model** called **SmileGuard** for what appears to be an expo/thesis project. Training crashed at epoch 77/200 with ~7% mAP. The root cause has been **fully diagnosed**: the dataset was corrupted during a manual merge by a teammate. A fix script has already been written. The user needs to run it, then retrain cleanly.

**Current status:** Ready to re-merge and retrain. Script is written. Waiting on execution.

---

## 📁 Project Structure (Local Machine)

```
C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train\
├── yolo_dataset_balanced\          ← CORRUPTED dataset (DO NOT USE)
│   ├── data.yaml
│   ├── train\
│   │   ├── images\
│   │   └── labels\                 ← ALL LABELS ARE BROKEN
│   └── val\
│       ├── images\
│       └── labels\
├── runs\detect\runs\detect\Dental_Detection\YOLOv8_Training1\
│   ├── args.yaml                   ← Training config (saved)
│   ├── results.csv                 ← 77 epochs of training logs
│   ├── labels.jpg                  ← Confirms label corruption
│   ├── train_batch0/1/2.jpg        ← Shows bounding box chaos
│   └── weights\
│       ├── best.pt                 ← Useless (trained on bad data)
│       └── last.pt                 ← Useless (trained on bad data)
├── merge_datasets.py               ← THE FIX SCRIPT (run this!)
└── smileguard_merged\              ← Will be created after running script
```

---

## 🎯 Target Classes (5 total)

| ID | Class Name | Notes |
|----|-----------|-------|
| 0 | `tooth` | Healthy tooth baseline |
| 1 | `caries` | Cavities / dental decay |
| 2 | `calculus` | Tartar buildup |
| 3 | `gingivitis` | Gum disease / inflammation |
| 4 | `tooth_discoloration` | Staining, fluorosis, etc. |

---

## 🔬 Root Cause Analysis — What Went Wrong

### The Smoking Gun
The `labels.jpg` diagnostic from the crashed training revealed:

- **x-y scatter plot**: All annotation centers clustered at exactly `(0.5, 0.5)`
- **width-height scatter plot**: All boxes sized at `(1.0, 1.0)`

This means **every single bounding box annotation covers the entire image** — the model had no spatial localization data whatsoever. It was learning class classification on full images, not object detection.

### What The Corrupted Labels Look Like
```
4 0.500000 0.500000 1.000000 1.000000
```
This is technically valid YOLO format (values are 0–1) but the coordinates are **semantically wrong** — it's a dummy box covering the whole image.

### Why This Happened
The user's teammate merged 6 Roboflow datasets using a custom script. During that process, the script likely:
1. Detected coordinate format issues between datasets
2. Instead of properly converting, it **replaced all coordinates with dummy values**
3. The original coordinate data is **not recoverable** from these label files

### Why mAP Was 7–8% After 77 Epochs
- Model losses WERE going down (box_loss: 1.84 → 1.09, cls_loss: 3.06 → 1.67) ✅
- Model was learning to classify dental conditions ✅
- Model had zero idea WHERE in the image to put boxes ❌
- mAP measures localization + classification → localization was 0 → mAP tank ❌

---

## 📊 Training Config That Was Used (args.yaml)

```yaml
model: yolov8m.pt           # Medium backbone — good choice, keep this
data: [CORRUPTED PATH]      # Will change to smileguard_merged/data.yaml
epochs: 200
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.01                   # Default YOLO LR — fine
lrf: 0.01
dropout: 0.15               # Some regularization — reasonable
patience: 15                # Early stopping after 15 epochs no improvement
warmup_epochs: 5
close_mosaic: 10
mosaic: 0.8
mixup: 0.05
copy_paste: 0.1
degrees: 10.0
device: '0'                 # GPU training
workers: 4
seed: 0
amp: true                   # Mixed precision — keep for speed
```

### What To Keep vs Change For Retraining

| Parameter | Previous | Recommended | Reason |
|-----------|----------|-------------|--------|
| `model` | yolov8m.pt | yolov8m.pt ✅ | Good balance of speed/accuracy |
| `epochs` | 200 | 200 ✅ | Fine with clean data |
| `lr0` | 0.01 | 0.01 ✅ | YOLO default, appropriate |
| `batch` | 16 | 16 ✅ | Fine unless VRAM limited |
| `patience` | 15 | 50 ⚠️ | 15 is too aggressive early on |
| `dropout` | 0.15 | 0.1 | Slight reduction with clean data |
| `imgsz` | 640 | 640 ✅ | Standard |

---

## 🗂️ The 6 Source Datasets

All from Roboflow, exported in YOLOv8 format:

| # | Name | Roboflow URL | Download Link |
|---|------|-------------|---------------|
| 1 | Dental_Caries | [Link](https://universe.roboflow.com/gfernandes/dental-caries-ngfuj/dataset/2) | `https://universe.roboflow.com/ds/tFyMSU1RVO?key=wc2bvncf1y` |
| 2 | Dental_Datasets | [Link](https://universe.roboflow.com/image-segmentation-ltmbq/dental-datasets-al4p8/dataset/3) | `https://universe.roboflow.com/ds/aDSosLihee?key=70ao7AXKb6` |
| 3 | Dental_Detecting | [Link](https://universe.roboflow.com/detection-b0acc/dental-detectin2/dataset/1) | `https://universe.roboflow.com/ds/Xj6tRZyxxt?key=S2c3Bh5WQT` |
| 4 | Dental_Detection | [Link](https://universe.roboflow.com/thesis-image-testing/dental-detection-5uclw/dataset/1) | `https://universe.roboflow.com/ds/yqisu9HuJm?key=4v2cMRnxk3` |
| 5 | Dental_Disease | [Link](https://universe.roboflow.com/dea-clc9x/dental-disease-ujstp/dataset/3) | `https://universe.roboflow.com/ds/MAH07gJupU?key=f3sfSOA0x9` |
| 6 | Dental_Mates | Same as #5 — **duplicate, skip** | Same URL as #5 |

> ⚠️ Note: Dental_Mates and Dental_Disease point to the same Roboflow project. The merge script already handles this by deduplicating filenames.

---

## 🔧 The Fix — merge_datasets.py

The script is already written and saved at:
```
C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train\merge_datasets.py
```

### What It Does
1. Downloads all 5 unique datasets from Roboflow directly
2. Reads each `data.yaml` to get source class names
3. Remaps all class names to the 5 unified target classes via `CLASS_MAP`
4. **Skips** any labels with full-image dummy boxes (`w >= 0.99 AND h >= 0.99`)
5. **Skips** any labels with coordinates outside `[0, 1]`
6. Handles filename collisions (renames duplicates instead of overwriting)
7. Maps `test` split → `val` split
8. Writes a clean `data.yaml` with absolute paths
9. Prints a full summary with per-split counts and any unmapped class warnings

### How To Run
```bash
# Install dependency
pip install pyyaml

# Run from SmileGuard-Train directory
cd C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train
python merge_datasets.py
```

### Expected Output
```
============================================================
  SmileGuard Dataset Merger
============================================================

[1] Downloading https://universe.roboflow.com/ds/tFyMSU1RVO...
    Downloaded: XX.X MB
    Extracting...
    Classes in dataset 1: ['tooth', 'caries', ...]
    Splits found: ['train', 'val']
...

============================================================
  MERGE COMPLETE
============================================================
  Train images : XXXX
  Val images   : XXXX
  Total instances : XXXXX
  Skipped (bad labels)  : XXXX
  Skipped (no image)    : XX

  Output folder: C:\...\smileguard_merged
  data.yaml written ✓

  All classes mapped successfully ✓
```

### ⚠️ If You See "UNMAPPED CLASSES"
The script will print something like:
```
⚠ UNMAPPED CLASSES (add to CLASS_MAP if needed):
  - 'dental_plaque'
  - 'crown'
```
This means some source classes didn't match the `CLASS_MAP`. You have two choices:
1. **Map them** → open `merge_datasets.py`, find `CLASS_MAP`, add entries like `"dental_plaque": "calculus"`
2. **Ignore them** → they'll be skipped. Only do this if the class is irrelevant to your 5 targets.

---

## 🚀 Retraining Command (After Merge)

```bash
yolo train \
  model=yolov8m.pt \
  data=C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train\smileguard_merged\data.yaml \
  epochs=200 \
  imgsz=640 \
  batch=16 \
  patience=50 \
  dropout=0.1 \
  optimizer=AdamW \
  device=0 \
  project=runs/detect/Dental_Detection \
  name=YOLOv8_Training2 \
  amp=True \
  seed=0
```

---

## 📈 What Healthy Training Should Look Like

### First 10 Epochs
| Metric | Expected Range |
|--------|---------------|
| `train/box_loss` | 2.0–3.5, dropping |
| `train/cls_loss` | 2.0–4.0, dropping |
| `metrics/mAP50(B)` | 0.05–0.25 (still warming up) |

### By Epoch 50
| Metric | Expected Range |
|--------|---------------|
| `train/box_loss` | ~1.0–1.5 |
| `metrics/mAP50(B)` | 0.30–0.55 |
| `metrics/precision(B)` | 0.40–0.65 |

### By Epoch 100–150 (Healthy Model)
| Metric | Expected Range |
|--------|---------------|
| `metrics/mAP50(B)` | 0.60–0.80 |
| `val/box_loss` | Close to train loss (not diverging) |

### 🚨 Red Flags During Training
- `mAP50` still below 0.10 after epoch 30 → label problem still exists
- `train/box_loss` dropping but `val/box_loss` rising → overfitting
- `metrics/precision` OR `metrics/recall` stuck near 0 → class mapping issue
- Only 300 instances per epoch → labels not loading (path issue in data.yaml)

---

## 🏗️ Dataset Statistics (From Corrupted Run — for Reference)

The class distribution was actually **well balanced**, which is good:

| Class | Instances |
|-------|-----------|
| tooth | 5,475 |
| calculus | 2,738 |
| caries | 2,509 |
| gingivitis | 2,502 |
| tooth_discoloration | 2,503 |
| **Total** | **~15,727** |

The distribution itself was fine — the coordinates were the problem, not the class counts.

---

## 💡 mAP Target Discussion

The user asked whether to target 95% mAP or stay at 80%. Here's the agreed-upon position:

- **With 12,000 images and 5 well-defined classes: 95% mAP is a legitimate target**
- The earlier concern about overfitting was based on assuming a small dataset — that concern is lower with 12k images
- **Key distinction**: mAP on the **validation set** is the real metric. Train mAP is irrelevant.
- A healthy model will have **train mAP and val mAP climbing together**
- If they diverge (train >> val), that's overfitting — not if val mAP is high

### Realistic Expectations
| Scenario | Expected Max mAP50 |
|----------|-------------------|
| Clean data, yolov8m, 200 epochs | 75–88% |
| Clean data + heavy augmentation | 82–92% |
| Clean data + larger backbone (yolov8l) | 85–93% |
| 95%+ | Possible but requires very tight annotations + large diverse data |

---

## 🔍 Diagnostic Tools

### Check If Labels Are Loading Correctly
```python
import os
from pathlib import Path

label_dir = r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train\smileguard_merged\train\labels"
files = list(Path(label_dir).glob("*.txt"))
non_empty = [f for f in files if f.stat().st_size > 0]
print(f"Total label files: {len(files)}")
print(f"Non-empty: {len(non_empty)}")

# Sample a label to verify format
with open(non_empty[0]) as f:
    print(f"\nSample label ({non_empty[0].name}):")
    print(f.read())
```

### Verify Bounding Box Sanity
```python
import os
from pathlib import Path
import statistics

label_dir = r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train\smileguard_merged\train\labels"
widths, heights = [], []

for lbl in Path(label_dir).glob("*.txt"):
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                widths.append(float(parts[3]))
                heights.append(float(parts[4]))

print(f"Avg box width:  {statistics.mean(widths):.3f}")
print(f"Avg box height: {statistics.mean(heights):.3f}")
print(f"Max box width:  {max(widths):.3f}")
print(f"Max box height: {max(heights):.3f}")
# Healthy values: avg width/height should be 0.05–0.35 for dental images
# BAD: avg close to 1.0 = still corrupted
```

### Quick YOLO Validation Run
```bash
# After merging, validate the dataset before training
yolo val model=yolov8m.pt data=smileguard_merged/data.yaml
```

---

## 🗺️ Class Remapping Logic

The `CLASS_MAP` in `merge_datasets.py` handles all known class name variants across the 6 datasets. Key mappings:

```python
# If a dataset calls it "cavity" → maps to "caries" (class ID 1)
# If a dataset calls it "tartar" → maps to "calculus" (class ID 2)
# If a dataset calls it "gum_disease" → maps to "gingivitis" (class ID 3)
# etc.
```

If new unmapped classes appear after running the script, the user will tell you what they are and you need to decide together: map them to an existing class, or skip them.

---

## 📋 Immediate Next Steps (In Order)

```
[ ] 1. Run merge_datasets.py locally
[ ] 2. Check the summary output — note train/val counts and any unmapped classes
[ ] 3. Run the bbox sanity check script above to confirm avg box size is reasonable
[ ] 4. Start retraining with the command above (YOLOv8_Training2)
[ ] 5. Monitor first 20 epochs — mAP should be climbing above 5% by epoch 10
[ ] 6. Check labels.jpg in new training run — should show scattered dots, not clustered
[ ] 7. Let training run to completion (200 epochs or early stop)
[ ] 8. Evaluate best.pt on test set
```

---

## 🧩 Technical Stack

| Component | Version/Detail |
|-----------|---------------|
| Model | YOLOv8m (ultralytics) |
| Framework | PyTorch + Ultralytics |
| Optimizer | AdamW |
| GPU | CUDA device 0 |
| OS | Windows (local machine) |
| Python | Standard env (pip) |
| Dataset format | YOLOv8 (normalized xywh) |
| Image size | 640x640 |

---

## 🙋 About The User

- Building SmileGuard for an **expo/thesis project**
- Has a teammate who handled dataset merging (the source of the bug)
- Comfortable with YOLO training concepts but learning the nuances
- Using VS Code with Claude extension for development
- Machine has a dedicated GPU (device='0' in training config)
- Storage on OneDrive

---

## ❓ Open Questions Not Yet Resolved

1. **How many total images are actually in the 6 source datasets?** We know the corrupted dataset showed ~12k images but we don't know the clean per-dataset breakdown.
2. **What unmapped class names will appear?** Won't know until the merge script runs.
3. **VRAM capacity?** Batch size of 16 at 640px with yolov8m is typically fine on 8GB+ VRAM. If OOM errors occur, drop batch to 8.
4. **Dental_Thesis dataset** (`https://universe.roboflow.com/caries-ccvwj/thesis-dental`) — this was mentioned but no download link was provided. User may want to add it manually to the `DATASET_URLS` list in merge_datasets.py.

---

## 📎 Files Already Created

| File | Location | Purpose |
|------|----------|---------|
| `merge_datasets.py` | SmileGuard-Train root | Downloads + merges all datasets cleanly |
| `labels_comparison.png` | (generated in session) | Visual showing bad vs good labels.jpg |
| `SMILEGUARD_HANDOFF.md` | This file | Full context for VS Code Claude |

---

*Handoff prepared by Claude (claude.ai) — March 11, 2026*
*Continue from Step 1 of "Immediate Next Steps" above.*
