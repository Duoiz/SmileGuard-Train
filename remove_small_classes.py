"""
Remove any class with fewer than 90 total annotations from yolo_dataset_merged.
Re-index remaining classes and update data.yaml.
"""
from pathlib import Path
from collections import defaultdict

BASE = Path(r"C:\Users\nangi\OneDrive\Documents\Code\Expo\SmileGuard-Train")
MERGED = BASE / "yolo_dataset_merged"

CLASSES = [
    'tooth', 'caries', 'cavity', 'crack',
    'calculus', 'gingivitis', 'hypodontia', 'mouth_ulcer', 'tooth_discoloration',
    'braces', 'misaligned_tooth', 'plaque', 'crown'
]

THRESHOLD = 90

# ── Step 1: Count total annotations per class ──
print("=" * 60)
print("Current class totals:")
print("=" * 60)

totals = defaultdict(int)
for split in ["train", "val", "test"]:
    for lbl in (MERGED / split / "labels").glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    totals[int(parts[0])] += 1

for cls_id in sorted(totals.keys()):
    name = CLASSES[cls_id]
    status = "KEEP" if totals[cls_id] >= THRESHOLD else "REMOVE"
    print(f"  {cls_id}={name}: {totals[cls_id]}  --> {status}")

# ── Step 2: Build keep list and remap ──
keep_classes = [i for i in range(len(CLASSES)) if totals.get(i, 0) >= THRESHOLD]
old_to_new = {old: new for new, old in enumerate(keep_classes)}
new_names = [CLASSES[i] for i in keep_classes]

removed = [CLASSES[i] for i in range(len(CLASSES)) if i not in keep_classes]
print(f"\nRemoving classes: {removed}")
print(f"New class list ({len(new_names)}): {new_names}")
print(f"Remap: {{{', '.join(f'{CLASSES[k]}:{old_to_new[k]}' for k in keep_classes)}}}")

# ── Step 3: Rewrite all label files ──
print("\n" + "=" * 60)
print("Rewriting labels...")
print("=" * 60)

removed_annotations = 0
emptied_files = 0

for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"
    img_dir = MERGED / split / "images"
    split_removed = 0
    split_emptied = 0

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        with open(lbl_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        new_lines = []
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id in old_to_new:
                parts[0] = str(old_to_new[cls_id])
                new_lines.append(" ".join(parts))
            else:
                split_removed += 1

        if new_lines:
            with open(lbl_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")
        else:
            # Empty label file — remove image and label
            lbl_file.unlink()
            stem = lbl_file.stem
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img_file = img_dir / (stem + ext)
                if img_file.exists():
                    img_file.unlink()
                    break
            split_emptied += 1

    print(f"  {split}: removed {split_removed} annotations, deleted {split_emptied} empty images")
    removed_annotations += split_removed
    emptied_files += split_emptied

print(f"\n  Total: removed {removed_annotations} annotations, deleted {emptied_files} empty images")

# ── Step 4: Update data.yaml ──
yaml_path = MERGED / "data.yaml"
with open(yaml_path, "w") as f:
    f.write("train: ./train/images\n")
    f.write("val: ./val/images\n")
    f.write("test: ./test/images\n\n")
    f.write(f"nc: {len(new_names)}\n")
    f.write(f"names: {new_names}\n")

# ── Step 5: Final verification ──
print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

grand = defaultdict(int)
for split in ["train", "val", "test"]:
    lbl_dir = MERGED / split / "labels"
    img_dir = MERGED / split / "images"
    counts = defaultdict(int)
    for lbl in lbl_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                p = line.strip().split()
                if p:
                    counts[int(p[0])] += 1
    n_img = len(list(img_dir.glob("*")))
    n_lbl = len(list(lbl_dir.glob("*.txt")))
    print(f"\n{split}: {n_img} images, {n_lbl} labels")
    for c in sorted(counts.keys()):
        print(f"  {c}={new_names[c]}: {counts[c]}")
        grand[c] += counts[c]

print(f"\nGRAND TOTALS:")
total = 0
for c in sorted(grand.keys()):
    print(f"  {c}={new_names[c]}: {grand[c]}")
    total += grand[c]
print(f"\n  Total annotations: {total}")
print("Done!")
