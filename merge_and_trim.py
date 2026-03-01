"""
Unified merge-and-trim pipeline.
Merges multiple source datasets into yolo_dataset_merged, trims
over-represented classes, removes under-represented classes, and
writes a clean data.yaml.

Replaces: merge_and_trim_caries.py, merge_and_trim_mates.py,
          merge_and_trim_teeth.py, merge_and_trim_roboflow.py,
          merge_and_trim_roboflow2.py, remove_small_classes.py
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

# ╔════════════════════════════════════════════════════════════╗
# ║                     CONFIGURATION                         ║
# ╚════════════════════════════════════════════════════════════╝

BASE = Path(r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train")
MERGED = BASE / "yolo_dataset_merged"

# Unified class list for the merged dataset
CLASSES = [
    "tooth",                # 0
    "caries",               # 1
    "cavity",               # 2
    "crack",                # 3
    "calculus",             # 4
    "gingivitis",           # 5
    "hypodontia",           # 6
    "mouth_ulcer",          # 7
    "tooth_discoloration",  # 8
]

# Split mapping: source folder name → merged folder name
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}

# Image extensions to try when matching labels → images
IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# ──────────────────────────────────────────────
# Source dataset definitions
# ──────────────────────────────────────────────
# Each source has:
#   name          - display name
#   path          - folder name under BASE
#   class_map     - {source_class_id: merged_class_id}  (unmapped IDs are dropped)
#   prefix        - collision prefix for filenames
#   require_class - (optional) only copy an image if it contains this merged class
#   skip_only     - (optional) list of source class IDs; images with ONLY these
#                   classes (and no mapped classes) are skipped entirely

SOURCES = [
    {
        "name": "Dental_Mates",
        "path": "Dental_Mates",
        "class_map": {
            1: 2,   # Cavity → cavity
            2: 3,   # Crack  → crack
            3: 0,   # Tooth  → tooth
        },
        "prefix": "dm_",
        "skip_only": [0],  # drop images that have ONLY Caries (src class 0)
    },
    {
        "name": "Dental_Caries",
        "path": "Dental_Caries",
        "class_map": {
            2: 3,   # Crack → crack
            3: 0,   # Tooth → tooth
        },
        "prefix": "dc_",
        "require_class": 3,  # only import images containing crack
    },
    {
        "name": "Dental_Teeth",
        "path": "Dental_Teeth",
        "class_map": {
            2: 3,   # Crack → crack
            3: 0,   # Tooth → tooth
        },
        "prefix": "dt_",
        "require_class": 3,  # only import images containing crack
    },
    {
        "name": "Roboflow-dataset",
        "path": "Roboflow-dataset",
        "class_map": {
            0: 1,   # Caries → caries
            1: 2,   # Cavity → cavity
            2: 3,   # Crack  → crack
            3: 0,   # Tooth  → tooth
        },
        "prefix": "rf_",
    },
    {
        "name": "Dental-Dataset-roboflow2",
        "path": "Dental-Dataset-roboflow2",
        "class_map": {
            2: 2,   # Cavity → cavity
        },
        "prefix": "r2_",
    },
]

# ──────────────────────────────────────────────
# Trim & prune thresholds
# ──────────────────────────────────────────────
# Per-class annotation caps (total across all splits).
# Classes exceeding these are trimmed by removing single-class-only images first.
TRIM_TARGETS = {
    0: 4000,   # tooth
    1: 4000,   # caries
}

# Split allocation for trimming
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Minimum total annotations to keep a class; below this → class is removed entirely
MIN_ANNOTATIONS = 90

# If True, wipe the merged dir before starting (full rebuild)
CLEAN_START = True


# ╔════════════════════════════════════════════════════════════╗
# ║                      HELPERS                              ║
# ╚════════════════════════════════════════════════════════════╝

def find_image(img_dir: Path, stem: str) -> Path | None:
    """Find an image file matching a label stem."""
    for ext in IMG_EXTS:
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def count_classes(labels_dir: Path) -> defaultdict:
    """Count annotations per class in a labels directory."""
    counts = defaultdict(int)
    for lbl_file in labels_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


def print_distribution(header: str):
    """Print per-split and total annotation counts."""
    print(f"\n{'=' * 60}")
    print(header)
    print("=" * 60)

    grand_total = defaultdict(int)
    for split in ["train", "val", "test"]:
        lbl_dir = MERGED / split / "labels"
        img_dir = MERGED / split / "images"
        if not lbl_dir.exists():
            continue
        counts = count_classes(lbl_dir)
        n_images = len(list(img_dir.glob("*")))
        n_labels = len(list(lbl_dir.glob("*.txt")))
        print(f"\n  {split}: {n_images} images, {n_labels} labels")
        for cls_id in sorted(counts.keys()):
            name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
            print(f"    {cls_id}={name}: {counts[cls_id]:,}")
            grand_total[cls_id] += counts[cls_id]

    print(f"\n  TOTALS:")
    for cls_id in sorted(grand_total.keys()):
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
        print(f"    {cls_id}={name}: {grand_total[cls_id]:,}")

    return grand_total


# ╔════════════════════════════════════════════════════════════╗
# ║               STEP 1: MERGE ALL SOURCES                   ║
# ╚════════════════════════════════════════════════════════════╝

def merge_all():
    if CLEAN_START and MERGED.exists():
        print("Cleaning previous merged dataset...")
        shutil.rmtree(MERGED)

    # Create output dirs
    for split in SPLIT_MAP.values():
        (MERGED / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("STEP 1: Merging all source datasets")
    print("=" * 60)

    grand_total = 0

    for src in SOURCES:
        src_dir = BASE / src["path"]
        class_map = src["class_map"]
        prefix = src["prefix"]
        require_class = src.get("require_class")  # merged class ID that must be present
        skip_only = set(src.get("skip_only", []))  # source class IDs

        print(f"\n  [{src['name']}] ({src_dir.name})")

        if not src_dir.exists():
            print(f"    WARNING: Directory not found, skipping")
            continue

        source_total = 0

        for rf_split, merged_split in SPLIT_MAP.items():
            rf_img_dir = src_dir / rf_split / "images"
            rf_lbl_dir = src_dir / rf_split / "labels"
            merged_img_dir = MERGED / merged_split / "images"
            merged_lbl_dir = MERGED / merged_split / "labels"

            if not rf_lbl_dir.exists():
                continue

            split_count = 0

            for lbl_file in sorted(rf_lbl_dir.glob("*.txt")):
                stem = lbl_file.stem
                img_src = find_image(rf_img_dir, stem)
                if img_src is None:
                    continue

                # ── Remap class IDs ──
                new_lines = []
                mapped_classes = set()     # merged class IDs present
                src_classes_only = set()   # source class IDs present

                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        old_cls = int(parts[0])
                        src_classes_only.add(old_cls)
                        if old_cls in class_map:
                            new_cls = class_map[old_cls]
                            parts[0] = str(new_cls)
                            new_lines.append(" ".join(parts))
                            mapped_classes.add(new_cls)

                # ── Apply filters ──
                # Skip if no usable annotations after remapping
                if not new_lines:
                    continue

                # skip_only: drop images where ALL source classes are in the skip set
                if skip_only and src_classes_only.issubset(skip_only | set(class_map.keys()) - set(class_map.keys())):
                    # More precise: drop if it only has skip_only source classes
                    # (i.e., no annotations survived except from skip_only)
                    non_skip_lines = [
                        l for l in new_lines
                        if True  # they survived class_map, so they're valid
                    ]
                    # Actually: if the image originally only had classes in skip_only
                    # (ones NOT in class_map), new_lines would be empty — already caught above.
                    # skip_only is for: source cls IS mapped but we still want to drop
                    # images that have ONLY that source cls.
                    pass  # The skip logic is already handled by class_map exclusion

                # require_class: only keep images containing this specific merged class
                if require_class is not None and require_class not in mapped_classes:
                    continue

                # ── Handle filename collision ──
                dst_img_name = img_src.name
                dst_lbl_name = stem + ".txt"
                if (merged_img_dir / dst_img_name).exists():
                    dst_img_name = prefix + dst_img_name
                    dst_lbl_name = prefix + dst_lbl_name

                # ── Write label + copy image ──
                with open(merged_lbl_dir / dst_lbl_name, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                shutil.copy2(img_src, merged_img_dir / dst_img_name)
                split_count += 1

            if split_count:
                print(f"    {rf_split:>5} → {merged_split}: {split_count:,} images")
            source_total += split_count

        print(f"    Total: {source_total:,} images")
        grand_total += source_total

    print(f"\n  Grand total merged: {grand_total:,} images")


# ╔════════════════════════════════════════════════════════════╗
# ║           STEP 2: TRIM OVER-REPRESENTED CLASSES           ║
# ╚════════════════════════════════════════════════════════════╝

def trim_classes():
    if not TRIM_TARGETS:
        print("\n  No trim targets configured, skipping.")
        return

    print(f"\n{'=' * 60}")
    print("STEP 2: Trimming over-represented classes")
    print("=" * 60)

    for cls_id, total_target in TRIM_TARGETS.items():
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
        print(f"\n  Trimming {cls_id}={cls_name} → target {total_target:,} total annotations")

        for split in ["train", "val", "test"]:
            lbl_dir = MERGED / split / "labels"
            img_dir = MERGED / split / "images"
            target = int(total_target * SPLIT_FRACTIONS[split])

            # Separate single-class-only files from mixed files
            only_files = []   # contain ONLY this class
            mixed_files = []  # contain this class + others

            for lbl_file in sorted(lbl_dir.glob("*.txt")):
                classes_in_file = set()
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes_in_file.add(int(parts[0]))
                if cls_id not in classes_in_file:
                    continue
                if classes_in_file == {cls_id}:
                    only_files.append(lbl_file)
                else:
                    mixed_files.append(lbl_file)

            # Count current annotations
            def count_cls(files):
                return sum(
                    1
                    for lbl in files
                    for line in open(lbl)
                    if line.strip().split() and int(line.strip().split()[0]) == cls_id
                )

            current = count_cls(only_files) + count_cls(mixed_files)
            print(f"    {split}: {current:,} annotations, target={target:,}")

            if current <= target:
                print(f"      Already at or below target.")
                continue

            excess = current - target
            random.shuffle(only_files)
            removed_annots, removed_files = 0, 0

            for lbl_file in only_files:
                if removed_annots >= excess:
                    break
                n = sum(
                    1 for line in open(lbl_file)
                    if line.strip().split() and int(line.strip().split()[0]) == cls_id
                )
                stem = lbl_file.stem
                lbl_file.unlink()
                img = find_image(img_dir, stem)
                if img:
                    img.unlink()
                removed_annots += n
                removed_files += 1

            print(f"      Removed {removed_files:,} images ({removed_annots:,} annotations)")


# ╔════════════════════════════════════════════════════════════╗
# ║         STEP 3: REMOVE UNDER-REPRESENTED CLASSES          ║
# ╚════════════════════════════════════════════════════════════╝

def prune_small_classes():
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Removing classes with < {MIN_ANNOTATIONS} total annotations")
    print("=" * 60)

    # Count totals across all splits
    totals = defaultdict(int)
    for split in ["train", "val", "test"]:
        lbl_dir = MERGED / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        totals[int(parts[0])] += 1

    # Decide which to keep
    keep_ids = sorted(cid for cid in range(len(CLASSES)) if totals.get(cid, 0) >= MIN_ANNOTATIONS)
    remove_ids = sorted(cid for cid in range(len(CLASSES)) if totals.get(cid, 0) < MIN_ANNOTATIONS and totals.get(cid, 0) > 0)

    if not remove_ids:
        print("  All classes meet the minimum threshold. Nothing to remove.")
        return CLASSES, {i: i for i in range(len(CLASSES))}

    removed_names = [CLASSES[i] for i in remove_ids]
    print(f"  Removing: {removed_names}")
    for cid in remove_ids:
        print(f"    {cid}={CLASSES[cid]}: {totals.get(cid, 0)} annotations")

    # Build remap: old_id → new_id (contiguous)
    old_to_new = {old: new for new, old in enumerate(keep_ids)}
    new_names = [CLASSES[i] for i in keep_ids]
    print(f"  New class list ({len(new_names)}): {new_names}")

    # Rewrite all label files
    remove_set = set(remove_ids)
    for split in ["train", "val", "test"]:
        lbl_dir = MERGED / split / "labels"
        img_dir = MERGED / split / "images"
        if not lbl_dir.exists():
            continue

        removed_files = 0
        rewritten = 0
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            new_lines = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls = int(parts[0])
                    if old_cls in remove_set:
                        continue
                    if old_cls in old_to_new:
                        parts[0] = str(old_to_new[old_cls])
                        new_lines.append(" ".join(parts))

            if not new_lines:
                # No annotations left — remove image+label
                stem = lbl_file.stem
                lbl_file.unlink()
                img = find_image(img_dir, stem)
                if img:
                    img.unlink()
                removed_files += 1
            else:
                with open(lbl_file, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                rewritten += 1

        print(f"  {split}: rewritten {rewritten:,}, removed {removed_files:,} empty-label images")

    return new_names, old_to_new


# ╔════════════════════════════════════════════════════════════╗
# ║                STEP 4: WRITE data.yaml                    ║
# ╚════════════════════════════════════════════════════════════╝

def write_data_yaml(class_names: list):
    yaml_path = MERGED / "data.yaml"
    abs_base = str(MERGED.resolve()).replace("\\", "/")
    with open(yaml_path, "w") as f:
        f.write(f"train: {abs_base}/train/images\n")
        f.write(f"val: {abs_base}/val/images\n")
        f.write(f"test: {abs_base}/test/images\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    print(f"\n  Wrote {yaml_path}")


# ╔════════════════════════════════════════════════════════════╗
# ║                         MAIN                              ║
# ╚════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Step 1 — Merge everything
    merge_all()
    print_distribution("After merge")

    # Step 2 — Trim over-represented classes
    trim_classes()
    print_distribution("After trimming")

    # Step 3 — Remove under-represented classes
    final_names, remap = prune_small_classes()

    # Step 4 — Final report + data.yaml
    totals = print_distribution("FINAL DISTRIBUTION")
    write_data_yaml(final_names)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
