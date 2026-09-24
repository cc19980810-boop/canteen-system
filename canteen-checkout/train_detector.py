#!/usr/bin/env python3
"""Train a class-agnostic food detector (one class: "food") with Ultralytics.

The detector only answers "where are the dishes and how many"; which dish it
is, is left to the retrieval recognizer. That way adding a new menu item never
requires retraining this model.

Examples
    python train_detector.py --data data/unimib_yolo/data.yaml                       # YOLO11s-seg
    python train_detector.py --data ... --model yolov8s-seg.pt --epochs 150 --batch 16
    python train_detector.py --data ... --model yolo11n.pt   # boxes only (labels from --task detect)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml written by tools/convert_unimib.py")
    ap.add_argument("--model", default="yolo11s-seg.pt",
                    help="yolo11{n,s,m}-seg.pt / yolov8{n,s,m}-seg.pt, or *.pt without -seg for boxes only")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto")
    ap.add_argument("--device", default=None, help="0 / 0,1 / cpu / mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/detector")
    ap.add_argument("--name", default="unimib")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        single_cls=True,       # everything is "food"; SKU identity comes from retrieval
        cos_lr=True,
        seed=0,
        # Augmentation tuned for top-down tray photos:
        degrees=15.0,          # trays are photographed at arbitrary rotation
        flipud=0.5,            # top-down view: vertical flip is realistic
        fliplr=0.5,
        hsv_h=0.005,           # keep hue almost fixed: colour is a key cue for food
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,             # blended dishes are not realistic
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    tuned = YOLO(str(best))
    report = {}
    for split in ("val", "test"):
        try:
            m = tuned.val(data=args.data, split=split, imgsz=args.imgsz, device=args.device,
                          project=args.project, name=f"{args.name}_eval_{split}")
            report[split] = {"box_map50": float(m.box.map50), "box_map50_95": float(m.box.map)}
            if getattr(m, "seg", None) is not None:
                report[split].update(mask_map50=float(m.seg.map50), mask_map50_95=float(m.seg.map))
        except Exception as e:  # noqa: BLE001 - test split may be absent
            report[split] = {"error": str(e)}
    print(json.dumps(report, indent=2))
    (Path(model.trainer.save_dir) / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
