"""
Merge Cavity + Crack + Tooth annotations from Dental_Mates into yolo_dataset_merged.
Caries (class 0) is skipped entirely — images that contain ONLY Caries are dropped.
Tooth is preserved in copied images to avoid false-background training, but trimmed to 4000 total.
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE = Path(r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train")
DENTAL_MATES = BASE / "Dental_Mates"
MERGED = BASE / "yolo_dataset_merged"

# ── Current merged classes (must match yolo_dataset_merged/data.yaml) ──
# Dental_Mates data.yaml: 0=Caries, 1=Cavity, 2=Crack, 3=Tooth
# Merged data.yaml:       0=tooth,  1=caries, 2=cavity, 3=crack, 4=calculus ...

CLASSES = [
    'tooth', 'caries', 'cavity', 'crack',
    'calculus', 'gingivitis', 'hypodontia', 'mouth_ulcer', 'tooth_discoloration',
]

# Import Cavity, Crack, and Tooth from Dental_Mates. Caries (0) is excluded.
# Images that contain ONLY Caries are dropped (new_lines will be empty).
MATES_TO_MERGED = {
    1: 2,   # Cavity → cavity
    2: 3,   # Crack  → crack
    3: 0,   # Tooth  → tooth  (kept to avoid unannotated-tooth confusion)
}

# Split mapping: Dental_Mates folder → merged folder
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

# ─────────────────────────────────────────────
# STEP 1: Merge Dental_Mates data with remapped IDs
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Merging Dental_Mates dataset into yolo_dataset_merged")
print("=" * 60)

total_copied = 0
for rf_split, merged_split in SPLIT_MAP.items():
    rf_img_dir = DENTAL_MATES / rf_split / "images"
    rf_lbl_dir = DENTAL_MATES / rf_split / "labels"
    merged_img_dir = MERGED / merged_split / "images"
    merged_lbl_dir = MERGED / merged_split / "labels"

    merged_img_dir.mkdir(parents=True, exist_ok=True)
    merged_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not rf_lbl_dir.exists():
        print(f"  Skipping {rf_split} — no labels dir")
        continue

    label_files = list(rf_lbl_dir.glob("*.txt"))
    split_count = 0

    for lbl_file in label_files:
        # Find corresponding image (try common extensions)
        stem = lbl_file.stem
        img_src = None
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            candidate = rf_img_dir / (stem + ext)
            if candidate.exists():
                img_src = candidate
                break

        if img_src is None:
            continue

        # Check for filename collision — prefix with "rf_" if needed
        dst_img_name = img_src.name
        dst_lbl_name = lbl_file.name
        if (merged_img_dir / dst_img_name).exists():
            dst_img_name = "rf_" + dst_img_name
            dst_lbl_name = "rf_" + dst_lbl_name

        # Remap label class IDs — keep Cavity/Crack/Tooth, drop Caries
        new_lines = []
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                old_cls = int(parts[0])
                if old_cls in MATES_TO_MERGED:
                    parts[0] = str(MATES_TO_MERGED[old_cls])
                    new_lines.append(" ".join(parts))

        # Skip images that have no usable annotations (e.g. Caries-only)
        if not new_lines:
            continue

        # Write remapped label
        with open(merged_lbl_dir / dst_lbl_name, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        # Copy image
        shutil.copy2(img_src, merged_img_dir / dst_img_name)
        split_count += 1

    total_copied += split_count
    print(f"  {rf_split} → {merged_split}: copied {split_count} images+labels")

print(f"\n  Total merged: {total_copied} images")

# ─────────────────────────────────────────────
# STEP 2: Count current class distribution
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Current class distribution (after merge)")
print("=" * 60)


def count_classes(labels_dir):
    """Count annotations per class in a labels directory."""
    counts = defaultdict(int)
    for lbl_file in Path(labels_dir).glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


total_counts = defaultdict(int)
for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"
    counts = count_classes(lbl_dir)
    print(f"\n  {split}:")
    for cls_id in sorted(counts.keys()):
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
        print(f"    {cls_id}={name}: {counts[cls_id]}")
        total_counts[cls_id] += counts[cls_id]

print(f"\n  TOTALS:")
for cls_id in sorted(total_counts.keys()):
    name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
    print(f"    {cls_id}={name}: {total_counts[cls_id]}")

# ─────────────────────────────────────────────
# STEP 3: Trim tooth (class 0) down to 4000 total
# Only tooth-only files are removed; mixed files (crack+tooth) are kept.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Trimming tooth class to ~4000 annotations")
print("=" * 60)

TARGET_TOOTH = 4000
TOOTH_CLS = 0

for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"
    img_dir = MERGED / split / "images"

    tooth_only_files = []
    mixed_files = []

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        classes_in_file = set()
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))
        if TOOTH_CLS not in classes_in_file:
            pass   # no tooth, skip
        elif classes_in_file == {TOOTH_CLS}:
            tooth_only_files.append(lbl_file)
        else:
            mixed_files.append(lbl_file)   # has tooth + other classes (e.g. crack)

    def count_tooth_in_files(files):
        total = 0
        for f in files:
            with open(f) as fh:
                for line in fh:
                    p = line.strip().split()
                    if p and p[0] == str(TOOTH_CLS):
                        total += 1
        return total

    tooth_in_mixed = count_tooth_in_files(mixed_files)
    tooth_in_only  = count_tooth_in_files(tooth_only_files)
    current_tooth  = tooth_in_mixed + tooth_in_only

    if split == "train":
        target = int(TARGET_TOOTH * 0.70)   # 2800
    elif split == "val":
        target = int(TARGET_TOOTH * 0.15)   # 600
    else:
        target = int(TARGET_TOOTH * 0.15)   # 600

    print(f"\n  {split}: tooth={current_tooth} (mixed={tooth_in_mixed}, only={tooth_in_only}), target={target}")

    if current_tooth <= target:
        print(f"    Already at or below target, no trimming needed.")
        continue

    excess = current_tooth - target
    print(f"    Need to remove ~{excess} tooth annotations")
    random.shuffle(tooth_only_files)
    removed_tooth, removed_files = 0, 0

    for lbl_file in tooth_only_files:
        if removed_tooth >= excess:
            break
        tooth_count = sum(
            1 for line in open(lbl_file)
            if line.strip().split() and line.strip().split()[0] == str(TOOTH_CLS)
        )
        stem = lbl_file.stem
        lbl_file.unlink()
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            img_file = img_dir / (stem + ext)
            if img_file.exists():
                img_file.unlink()
                break
        removed_tooth += tooth_count
        removed_files += 1

    print(f"    Removed {removed_files} tooth-only images ({removed_tooth} tooth annotations)")

# ─────────────────────────────────────────────
# STEP 4: Final counts & update data.yaml
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Final class distribution")
print("=" * 60)

final_total = defaultdict(int)
for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"
    img_dir = MERGED / split / "images"
    counts = count_classes(lbl_dir)
    n_images = len(list(img_dir.glob("*")))
    n_labels = len(list(lbl_dir.glob("*.txt")))
    print(f"\n  {split}: {n_images} images, {n_labels} labels")
    for cls_id in sorted(counts.keys()):
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
        print(f"    {cls_id}={name}: {counts[cls_id]}")
        final_total[cls_id] += counts[cls_id]

print(f"\n  GRAND TOTALS:")
for cls_id in sorted(final_total.keys()):
    name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
    print(f"    {cls_id}={name}: {final_total[cls_id]}")

# Update data.yaml
yaml_path = MERGED / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: ./train/images\n")
    f.write(f"val: ./val/images\n")
    f.write(f"test: ./test/images\n")
    f.write(f"\n")
    f.write(f"nc: {len(CLASSES)}\n")
    f.write(f"names: {CLASSES}\n")

print(f"\n  Updated {yaml_path}")
print("\nDone!")
