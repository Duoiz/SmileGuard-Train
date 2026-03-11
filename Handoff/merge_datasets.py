"""
SmileGuard Dataset Merger
=========================
Downloads all Roboflow datasets, remaps class names,
merges into a single clean YOLOv8-ready dataset.

Run: python merge_datasets.py
"""

import os
import shutil
import zipfile
import urllib.request
import yaml
import random
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "smileguard_merged"

DATASET_URLS = [
    "https://universe.roboflow.com/ds/tFyMSU1RVO?key=wc2bvncf1y",
    "https://universe.roboflow.com/ds/aDSosLihee?key=70ao7AXKb6",
    "https://universe.roboflow.com/ds/Xj6tRZyxxt?key=S2c3Bh5WQT",
    "https://universe.roboflow.com/ds/yqisu9HuJm?key=4v2cMRnxk3",
    "https://universe.roboflow.com/ds/MAH07gJupU?key=f3sfSOA0x9",
]

# Target unified classes (final dataset will use these IDs)
TARGET_CLASSES = [
    "tooth",
    "caries",
    "calculus",
    "gingivitis",
    "tooth_discoloration",
]

# Map every possible Roboflow class name → target class
# Add more mappings here if you discover new class names after inspection
CLASS_MAP = {
    # tooth
    "tooth": "tooth",
    "teeth": "tooth",
    "Tooth": "tooth",
    "Teeth": "tooth",
    "tooth_healthy": "tooth",
    "healthy": "tooth",

    # caries
    "caries": "caries",
    "Caries": "caries",
    "cavity": "caries",
    "Cavity": "caries",
    "dental_caries": "caries",
    "tooth_caries": "caries",
    "carries": "caries",   # common typo
    "carie": "caries",
    "decay": "caries",
    "dental decay": "caries",

    # calculus
    "calculus": "calculus",
    "Calculus": "calculus",
    "tartar": "calculus",
    "Tartar": "calculus",
    "dental_calculus": "calculus",
    "plaque": "calculus",

    # gingivitis
    "gingivitis": "gingivitis",
    "Gingivitis": "gingivitis",
    "gum_disease": "gingivitis",
    "gum disease": "gingivitis",
    "gingival": "gingivitis",
    "periodontitis": "gingivitis",
    "gum": "gingivitis",

    # tooth_discoloration
    "tooth_discoloration": "tooth_discoloration",
    "discoloration": "tooth_discoloration",
    "Discoloration": "tooth_discoloration",
    "stain": "tooth_discoloration",
    "Stain": "tooth_discoloration",
    "tooth_stain": "tooth_discoloration",
    "fluorosis": "tooth_discoloration",
    "hypomineralization": "tooth_discoloration",
}

TARGET_CLASS_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def download_and_extract(url, dest_dir, index):
    zip_path = os.path.join(dest_dir, f"dataset_{index}.zip")
    extract_path = os.path.join(dest_dir, f"dataset_{index}")

    print(f"\n[{index}] Downloading {url[:60]}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"    Downloaded: {os.path.getsize(zip_path) / 1e6:.1f} MB")
    except Exception as e:
        print(f"    ERROR downloading: {e}")
        return None

    print(f"    Extracting...")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)

    os.remove(zip_path)
    return extract_path


def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_yaml(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "data.yaml":
                return os.path.join(root, f)
    return None


def find_split_dirs(base_dir):
    """Return dict of split -> (images_dir, labels_dir)"""
    splits = {}
    for split in ["train", "valid", "val", "test"]:
        for root, dirs, files in os.walk(base_dir):
            if os.path.basename(root) == split:
                img_dir = os.path.join(root, "images")
                lbl_dir = os.path.join(root, "labels")
                if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
                    key = "val" if split in ("valid", "val") else split
                    splits[key] = (img_dir, lbl_dir)
    return splits


def validate_and_remap_label(label_path, src_classes, unmapped_log):
    """
    Read a YOLO label file, remap class IDs to target classes.
    Returns list of valid remapped lines, or None if file should be skipped.
    """
    remapped = []
    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed lines

        try:
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            continue

        # Validate coordinates are normalized
        if not all(0.0 <= c <= 1.0 for c in coords):
            continue  # skip broken coordinates

        # Check for full-image dummy boxes
        x_c, y_c, w, h = coords
        if w >= 0.99 and h >= 0.99:
            continue  # skip dummy full-image boxes

        # Remap class
        if cls_id >= len(src_classes):
            continue
        src_class_name = src_classes[cls_id]
        target_class = CLASS_MAP.get(src_class_name)

        if target_class is None:
            unmapped_log.add(src_class_name)
            continue  # skip unmapped classes

        target_id = TARGET_CLASS_ID[target_class]
        remapped.append(f"{target_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return remapped if remapped else None


def safe_filename(existing_names, stem, suffix):
    """Generate unique filename if collision exists."""
    name = f"{stem}{suffix}"
    if name not in existing_names:
        existing_names.add(name)
        return name
    counter = 1
    while True:
        name = f"{stem}_{counter}{suffix}"
        if name not in existing_names:
            existing_names.add(name)
            return name
        counter += 1


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SmileGuard Dataset Merger")
    print("=" * 60)

    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Output folders
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    stats = defaultdict(int)
    unmapped_classes = set()
    existing_filenames = {"train": set(), "val": set()}

    for idx, url in enumerate(DATASET_URLS, start=1):
        extract_path = download_and_extract(url, tmp_dir, idx)
        if extract_path is None:
            continue

        yaml_path = find_yaml(extract_path)
        if yaml_path is None:
            print(f"    WARNING: No data.yaml found in dataset {idx}, skipping.")
            continue

        cfg = read_yaml(yaml_path)
        src_classes = cfg.get("names", [])
        print(f"    Classes in dataset {idx}: {src_classes}")

        splits = find_split_dirs(extract_path)
        print(f"    Splits found: {list(splits.keys())}")

        for split, (img_dir, lbl_dir) in splits.items():
            # Map test → val
            out_split = "val" if split == "test" else split
            if out_split not in ("train", "val"):
                out_split = "train"

            out_img_dir = os.path.join(OUTPUT_DIR, out_split, "images")
            out_lbl_dir = os.path.join(OUTPUT_DIR, out_split, "labels")

            label_files = list(Path(lbl_dir).glob("*.txt"))

            for lbl_path in label_files:
                remapped = validate_and_remap_label(
                    str(lbl_path), src_classes, unmapped_classes
                )
                if remapped is None:
                    stats["skipped"] += 1
                    continue

                # Find matching image
                stem = lbl_path.stem
                img_path = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    candidate = os.path.join(img_dir, stem + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                if img_path is None:
                    stats["no_image"] += 1
                    continue

                # Safe unique filename
                img_ext = Path(img_path).suffix
                new_name = safe_filename(existing_filenames[out_split], stem, img_ext)
                new_stem = Path(new_name).stem

                # Copy image
                shutil.copy2(img_path, os.path.join(out_img_dir, new_name))

                # Write remapped label
                lbl_out = os.path.join(out_lbl_dir, new_stem + ".txt")
                with open(lbl_out, 'w') as f:
                    f.write("\n".join(remapped) + "\n")

                stats[f"{out_split}_added"] += 1
                stats["total_instances"] += len(remapped)

    # Write data.yaml
    data_yaml = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Cleanup tmp
    shutil.rmtree(tmp_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    print(f"  Train images : {stats['train_added']}")
    print(f"  Val images   : {stats['val_added']}")
    print(f"  Total instances : {stats['total_instances']}")
    print(f"  Skipped (bad labels)  : {stats['skipped']}")
    print(f"  Skipped (no image)    : {stats['no_image']}")
    print(f"\n  Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  data.yaml written ✓")

    if unmapped_classes:
        print(f"\n  ⚠ UNMAPPED CLASSES (add to CLASS_MAP if needed):")
        for c in sorted(unmapped_classes):
            print(f"    - '{c}'")
    else:
        print("\n  All classes mapped successfully ✓")

    print("\n  Next step — retrain with:")
    print(f"  yolo train model=yolov8m.pt data={os.path.abspath(OUTPUT_DIR)}/data.yaml epochs=200 imgsz=640 batch=16")


if __name__ == "__main__":
    main()
