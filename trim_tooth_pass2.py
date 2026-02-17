"""
Second pass: Strip tooth (class 0) annotations from mixed-class files
to bring total tooth count down to ~4000.
Keeps the images and all non-tooth annotations.
"""
import os
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE = Path(r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train")
MERGED = BASE / "yolo_dataset_merged"

TARGET_TOOTH = 4000
TOOTH_CLS = "0"

for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"

    if split == "train":
        target = int(TARGET_TOOTH * 0.70)   # 2800
    elif split == "val":
        target = int(TARGET_TOOTH * 0.15)   # 600
    else:
        target = int(TARGET_TOOTH * 0.15)   # 600

    # Find all mixed files (have tooth + other classes)
    mixed_files = []
    current_tooth = 0

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        with open(lbl_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        classes = set(l.split()[0] for l in lines)
        tooth_count = sum(1 for l in lines if l.split()[0] == TOOTH_CLS)
        current_tooth += tooth_count
        if TOOTH_CLS in classes and len(classes) > 1:
            mixed_files.append((lbl_file, tooth_count))

    print(f"{split}: current tooth={current_tooth}, target={target}")

    if current_tooth <= target:
        print(f"  Already at or below target.\n")
        continue

    excess = current_tooth - target
    print(f"  Need to strip ~{excess} tooth annotations from mixed files")

    # Shuffle and strip tooth lines from mixed files until we reach target
    random.shuffle(mixed_files)
    stripped = 0

    for lbl_file, tooth_count in mixed_files:
        if stripped >= excess:
            break

        with open(lbl_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        # Keep only non-tooth lines
        non_tooth_lines = [l for l in lines if l.split()[0] != TOOTH_CLS]

        if non_tooth_lines:  # Only rewrite if there are other annotations left
            with open(lbl_file, "w") as f:
                f.write("\n".join(non_tooth_lines) + "\n")
            stripped += tooth_count

    print(f"  Stripped {stripped} tooth annotations from mixed files\n")

# Final verification
print("=" * 60)
print("FINAL CLASS DISTRIBUTION")
print("=" * 60)

CLASSES = [
    'tooth', 'caries', 'cavity', 'crack',
    'calculus', 'gingivitis', 'hypodontia', 'mouth_ulcer', 'tooth_discoloration',
    'braces', 'misaligned_tooth', 'plaque', 'crown'
]

grand_total = defaultdict(int)
for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"
    img_dir = MERGED / split / "images"
    counts = defaultdict(int)
    for lbl_file in lbl_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1

    n_img = len(list(img_dir.glob("*")))
    n_lbl = len(list(lbl_dir.glob("*.txt")))
    print(f"\n{split}: {n_img} images, {n_lbl} labels")
    for cls_id in sorted(counts.keys()):
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
        print(f"  {cls_id}={name}: {counts[cls_id]}")
        grand_total[cls_id] += counts[cls_id]

print(f"\nGRAND TOTALS:")
total_annotations = 0
for cls_id in sorted(grand_total.keys()):
    name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
    print(f"  {cls_id}={name}: {grand_total[cls_id]}")
    total_annotations += grand_total[cls_id]
print(f"\n  Total annotations: {total_annotations}")
print("Done!")
