#!/usr/bin/env python3
"""Run the checkout on one tray photo (or a folder) and print the receipt.

    python demo.py --config configs/checkout.yaml --image tray.jpg --weight 540 --save out.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkout.pipeline import CheckoutPipeline, draw_receipt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/checkout.yaml")
    ap.add_argument("--image", required=True, type=Path, help="image or folder")
    ap.add_argument("--weight", type=float, default=None, help="measured tray weight in grams (optional)")
    ap.add_argument("--save", type=Path, default=None, help="annotated image (or folder when --image is a folder)")
    ap.add_argument("--json", action="store_true", help="print the receipt as JSON")
    args = ap.parse_args()

    pipe = CheckoutPipeline.from_config(args.config)
    files = sorted(p for p in args.image.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if args.image.is_dir() else [args.image]
    for f in files:
        img = cv2.imread(str(f))
        rec = pipe(img, measured_weight_g=args.weight)
        if args.json:
            print(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"== {f.name}")
            for i, it in enumerate(rec.items, 1):
                if it.status == "accept":
                    price = f"{it.price:.2f}" if it.price is not None else "  ?  "
                    print(f"  {i:2d}. {it.display_name:<28s} {price:>7s}   (sim {it.sim:.2f})")
                elif it.status == "uncertain":
                    opts = ", ".join(f"{c} {s:.2f}" for c, s in it.candidates)
                    print(f"  {i:2d}. ?? please choose: {opts}")
                else:
                    print(f"  {i:2d}. unrecognised item (best sim {it.sim:.2f})")
            print(f"  TOTAL {rec.total:.2f}" + (f"   expected weight ~{rec.expected_weight_g:.0f} g"
                                                if rec.expected_weight_g else ""))
            if rec.needs_review:
                print("  REVIEW: " + "; ".join(rec.review_reasons))
        if args.save:
            out = args.save / f.name if args.image.is_dir() else args.save
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), draw_receipt(img, rec))


if __name__ == "__main__":
    main()
