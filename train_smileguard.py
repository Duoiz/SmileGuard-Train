"""
SmileGuard Training Script — v2
================================
Clean merged dataset, GPU-accelerated, auto batch size.
"""

from ultralytics import YOLO
import multiprocessing


def main():
    model = YOLO("yolov8s.pt")  # Small model — fits 3GB VRAM better

    results = model.train(
        data="smileguard_merged/data.yaml",
        epochs=200,
        imgsz=640,
        batch=-1,          # Auto-detect optimal batch for available VRAM
        patience=50,
        dropout=0.1,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        device=0,
        workers=2,         # Reduced for Windows stability
        seed=0,
        amp=True,          # Mixed precision — saves VRAM
        warmup_epochs=5,
        close_mosaic=10,
        mosaic=0.8,
        mixup=0.05,
        copy_paste=0.1,
        degrees=10.0,
        project="runs/detect",
        name="SmileGuard_v2",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
