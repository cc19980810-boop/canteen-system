#!/usr/bin/env python3
"""Build (or extend) the retrieval gallery from folders of reference crops.

    crops/<class_name>/*.jpg   ->   gallery.npz

Examples
    # gallery from the UNIMIB train crops, DINOv2 features
    python build_gallery.py --crops data/unimib_yolo/crops/train --out gallery.npz

    # with the ArcFace-finetuned backbone
    python build_gallery.py --crops ... --checkpoint runs/embedder/best.pt --out gallery_ft.npz

    # add a new dish from a handful of photos, no retraining
    python build_gallery.py --append gallery.npz --crops new_dishes/ --out gallery.npz

Photos for --append can be whole tray pictures of a single dish: pass
--detector to crop them with the trained detector (largest detection kept),
so they are cropped exactly like runtime queries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkout.crop import crop_instance  # noqa: E402
from checkout.embedder import build_embedder  # noqa: E402
from checkout.index import VectorIndex  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, type=Path, help="folder with one sub-folder per class")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--append", type=Path, help="existing gallery.npz to extend")
    ap.add_argument("--embedder", default="timm", choices=["timm", "hist"])
    ap.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--checkpoint", help="fine-tuned embedder checkpoint (finetune_embedder.py)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all")
    ap.add_argument("--detector", help="YOLO weights: crop raw photos with the detector first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.embedder == "hist":
        emb_cfg = {"type": "hist"}
    else:
        emb_cfg = {"type": "timm", "model_name": args.model_name, "checkpoint": args.checkpoint,
                   "img_size": args.img_size, "device": args.device}
    embedder = build_embedder(emb_cfg)

    detector = None
    if args.detector:
        from checkout.detector import YoloDetector
        detector = YoloDetector(args.detector, conf=0.25)

    rng = np.random.default_rng(args.seed)
    class_dirs = sorted(d for d in args.crops.iterdir() if d.is_dir())
    if not class_dirs:
        sys.exit(f"no class folders in {args.crops}")

    index = VectorIndex.load(args.append) if args.append else None
    if index is not None:
        stored = {k: v for k, v in index.meta.get("embedder", {}).items() if k != "device"}
        wanted = {k: v for k, v in emb_cfg.items() if k != "device"}
        if stored and stored != wanted:
            sys.exit(f"embedder mismatch: gallery was built with {stored}, now {wanted}")

    all_vecs, all_lbls, all_src, names = [], [], [], []
    for d in class_dirs:
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if args.max_per_class and len(files) > args.max_per_class:
            files = [files[i] for i in sorted(rng.choice(len(files), args.max_per_class, replace=False))]
        crops, srcs = [], []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            if detector is not None:
                dets = detector(img)
                if not dets:
                    print(f"  skip {f.name}: nothing detected")
                    continue
                det = max(dets, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
                img = crop_instance(img, det.box, det.polygon)
            crops.append(img)
            srcs.append(str(f))
        if not crops:
            continue
        vecs = embedder(crops)
        if index is not None:
            index.add(vecs, d.name, srcs)
        else:
            names.append(d.name)
            all_vecs.append(vecs)
            all_lbls.append(np.full(len(vecs), len(names) - 1))
            all_src += srcs
        print(f"  {d.name}: {len(crops)}")

    if index is None:
        index = VectorIndex(np.concatenate(all_vecs), np.concatenate(all_lbls), names, all_src,
                            meta={"embedder": {k: v for k, v in emb_cfg.items()}})
    index.save(args.out)
    print(index.summary())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
