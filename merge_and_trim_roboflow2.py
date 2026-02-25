"""
Merge Roboflow dataset into yolo_dataset_merged and trim tooth class to ~4k.
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE = Path(r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train")
ROBOFLOW = BASE / "Dental-Dataset-roboflow2"
MERGED = BASE / "yolo_dataset_merged"

# ── Current merged classes ──
# 0=tooth, 1=caries, 2=cavity, 3=crack,
# 4=calculus, 5=gingivitis, 6=hypodontia, 7=mouth_ulcer, 8=tooth_discoloration
# 9=braces, 10=misaligned_tooth  (added by this script)

NEW_CLASSES = [
    'tooth', 'caries', 'cavity', 'crack',
    'calculus', 'gingivitis', 'hypodontia', 'mouth_ulcer', 'tooth_discoloration',
    'braces', 'misaligned_tooth',
]

# Dental-Dataset-roboflow class ID → merged class ID
# Dental-Dataset-roboflow: 0=Braces, 1=Caries, 2=Cavity, 3=Misaligned tooth, 4=Plaque, 5=crown
# Plaque and crown are excluded (not in target class set)
ROBOFLOW_TO_MERGED = {
    2: 2,   # Cavity           → cavity
    9: 9,   # Braces           → braces
    10: 10,  # Misaligned tooth → misaligned_tooth
}

# Split mapping: Roboflow folder → merged folder
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

# ─────────────────────────────────────────────
# STEP 1: Merge Roboflow data with remapped IDs
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Merging Roboflow dataset into yolo_dataset_merged")
print("=" * 60)

total_copied = 0
for rf_split, merged_split in SPLIT_MAP.items():
    rf_img_dir = ROBOFLOW / rf_split / "images"
    rf_lbl_dir = ROBOFLOW / rf_split / "labels"
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

        # Remap label class IDs
        new_lines = []
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                old_cls = int(parts[0])
                if old_cls in ROBOFLOW_TO_MERGED:
                    parts[0] = str(ROBOFLOW_TO_MERGED[old_cls])
                    new_lines.append(" ".join(parts))

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
        name = NEW_CLASSES[cls_id] if cls_id < len(NEW_CLASSES) else f"unknown_{cls_id}"
        print(f"    {cls_id}={name}: {counts[cls_id]}")
        total_counts[cls_id] += counts[cls_id]

print(f"\n  TOTALS:")
for cls_id in sorted(total_counts.keys()):
    name = NEW_CLASSES[cls_id] if cls_id < len(NEW_CLASSES) else f"unknown_{cls_id}"
    print(f"    {cls_id}={name}: {total_counts[cls_id]}")

# ─────────────────────────────────────────────
# STEP 3: Trim over-represented classes
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Trimming over-represented classes")
print("=" * 60)

# Per-class annotation caps across the whole dataset (train+val+test combined)
# Splits are allocated 70% train / 15% val / 15% test
TRIM_TARGETS = {
    0:  4000,   # tooth            — was ~22,731 total (grows further after merge)
    1:  4000,   # caries           — was ~8,572 total
    10: 4000,   # misaligned_tooth — was ~22,740 total
}

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def trim_class(cls_id, total_target):
    """Remove single-class images for cls_id until total annotations ≤ total_target."""
    cls_name = NEW_CLASSES[cls_id] if cls_id < len(NEW_CLASSES) else f"class_{cls_id}"
    print(f"\n  Trimming class {cls_id} ({cls_name}) → target {total_target} total")

    for split in ["train", "val", "test"]:
        lbl_dir = MERGED / split / "labels"
        img_dir = MERGED / split / "images"
        target = int(total_target * SPLIT_FRACTIONS[split])

        # Separate files that contain only this class vs mixed
        only_files = []
        mixed_files = []
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            classes_in_file = set()
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes_in_file.add(int(parts[0]))
            if cls_id not in classes_in_file:
                continue
            elif classes_in_file == {cls_id}:
                only_files.append(lbl_file)
            else:
                mixed_files.append(lbl_file)

        # Count current annotations for this class
        def count_cls_annots(files):
            total = 0
            for lbl_file in files:
                with open(lbl_file) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if parts and int(parts[0]) == cls_id:
                            total += 1
            return total

        current = count_cls_annots(only_files) + count_cls_annots(mixed_files)
        print(f"    {split}: current={current}, target={target}")

        if current <= target:
            print(f"      Already at or below target, skipping.")
            continue

        excess = current - target
        print(f"      Need to remove ~{excess} annotations")

        random.shuffle(only_files)
        removed_annots = 0
        removed_files = 0
        for lbl_file in only_files:
            if removed_annots >= excess:
                break
            cls_count = sum(
                1 for line in open(lbl_file)
                if line.strip().split() and int(line.strip().split()[0]) == cls_id
            )
            stem = lbl_file.stem
            lbl_file.unlink()
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img_file = img_dir / (stem + ext)
                if img_file.exists():
                    img_file.unlink()
                    break
            removed_annots += cls_count
            removed_files += 1

        print(f"      Removed {removed_files} images ({removed_annots} annotations)")


for cls_id, total_target in TRIM_TARGETS.items():
    trim_class(cls_id, total_target)

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
        name = NEW_CLASSES[cls_id] if cls_id < len(NEW_CLASSES) else f"unknown_{cls_id}"
        print(f"    {cls_id}={name}: {counts[cls_id]}")
        final_total[cls_id] += counts[cls_id]

print(f"\n  GRAND TOTALS:")
for cls_id in sorted(final_total.keys()):
    name = NEW_CLASSES[cls_id] if cls_id < len(NEW_CLASSES) else f"unknown_{cls_id}"
    print(f"    {cls_id}={name}: {final_total[cls_id]}")

# Update data.yaml
yaml_path = MERGED / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: ./train/images\n")
    f.write(f"val: ./val/images\n")
    f.write(f"test: ./test/images\n")
    f.write(f"\n")
    f.write(f"nc: {len(NEW_CLASSES)}\n")
    f.write(f"names: {NEW_CLASSES}\n")

print(f"\n  Updated {yaml_path}")
print("\nDone!")
