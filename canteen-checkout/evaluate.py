#!/usr/bin/env python3
"""Evaluate detector, recognizer and the full checkout on a converted split.

Reports three levels:
  detection     box precision / recall @IoU0.5, exact-count accuracy
  recognition   on ground-truth crops (detector errors excluded): top-1/top-3,
                accept rate, precision of accepted items, confusions
  end-to-end    per tray: auto-correct rate, and the SILENT ERROR rate =
                receipts that were wrong but NOT flagged for review. For a
                checkout this is the number that matters most (and the one a
                red-team exercise tries to push up).

--calibrate picks accept_sim / margin on a split (use val, not test) so that
accepted items reach a target precision while accepting as many as possible.

    python evaluate.py --dataset data/unimib_yolo --gallery gallery.npz \
        --detector runs/detector/unimib/weights/best.pt --split test
    python evaluate.py --dataset data/unimib_yolo --gallery gallery.npz \
        --detector oracle --split val --calibrate 0.98
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkout.boxes import greedy_match  # noqa: E402
from checkout.detector import Detection, OracleDetector  # noqa: E402
from checkout.embedder import build_embedder  # noqa: E402
from checkout.index import VectorIndex  # noqa: E402
from checkout.pipeline import CheckoutPipeline  # noqa: E402


def calibrate(top1, margin, correct, target):
    top1, margin, correct = map(np.asarray, (top1, margin, correct))
    best = None
    for a in np.round(np.arange(0.2, 0.96, 0.01), 2):
        for m in np.round(np.arange(0.0, 0.21, 0.01), 2):
            acc = (top1 >= a) & (margin >= m)
            if acc.sum() == 0:
                continue
            prec = correct[acc].mean()
            if prec >= target and (best is None or acc.mean() > best["accept_rate"] + 1e-9):
                best = {"accept_sim": float(a), "margin": float(m),
                        "accept_rate": float(acc.mean()), "precision": float(prec)}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="output folder of convert_unimib.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--detector", default="oracle", help="YOLO weights, or 'oracle' for GT boxes")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--embedder", default=None, choices=["timm", "hist"],
                    help="default: whatever the gallery was built with")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--accept-sim", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--unknown-sim", type=float, default=0.35)
    ap.add_argument("--calibrate", type=float, default=None, metavar="PRECISION")
    ap.add_argument("--out", type=Path, default=None, help="write JSON report here")
    ap.add_argument("--save-vis", type=Path, default=None, help="save annotated trays here")
    args = ap.parse_args()

    instances = json.loads((args.dataset / "instances.json").read_text())
    by_file = defaultdict(list)
    for i in instances:
        if i["split"] == args.split:
            by_file[i["file"]].append(i)
    if not by_file:
        sys.exit(f"no instances in split {args.split!r}")

    index = VectorIndex.load(args.gallery)
    emb_cfg = dict(index.meta.get("embedder", {"type": "timm"}))
    if args.embedder:
        emb_cfg["type"] = args.embedder
    for k in ("model_name", "checkpoint", "device"):
        if getattr(args, k) is not None:
            emb_cfg[k] = getattr(args, k)
    embedder = build_embedder(emb_cfg)
    print(f"gallery: {index.summary()}")
    known = set(index.class_names)

    oracle = OracleDetector(by_file)
    if args.detector == "oracle":
        detector = oracle
    else:
        from checkout.detector import YoloDetector
        detector = YoloDetector(args.detector, conf=args.conf, imgsz=args.imgsz, tta=args.tta)
    pipe = CheckoutPipeline(detector, embedder, index, accept_sim=args.accept_sim,
                            margin=args.margin, unknown_sim=args.unknown_sim)

    rec_rows = []   # (gt, top1, sim, margin, status, cands)
    det_tp = det_fp = det_fn = 0
    count_ok = 0
    e2e = Counter()
    if args.save_vis:
        args.save_vis.mkdir(parents=True, exist_ok=True)
    for f, gts in sorted(by_file.items()):
        img = cv2.imread(str(args.dataset / f))
        oracle.set_image(f)
        # --- recognition on GT crops
        gt_dets = [Detection(np.array(g["box"]), 1.0, np.array(g["polygon"])) for g in gts]
        for g, dec in zip(gts, pipe.recognize(img, gt_dets)):
            rec_rows.append((g["class"], dec.candidates[0][0], dec.score, dec.margin, dec.status,
                             [c for c, _ in dec.candidates]))
        # --- full pipeline
        receipt = pipe(img)
        pred_boxes = np.array([it.box for it in receipt.items]).reshape(-1, 4)
        gt_boxes = np.array([g["box"] for g in gts]).reshape(-1, 4)
        pairs = greedy_match(pred_boxes, gt_boxes, 0.5)
        det_tp += len(pairs)
        det_fp += len(pred_boxes) - len(pairs)
        det_fn += len(gt_boxes) - len(pairs)
        count_ok += int(len(pred_boxes) == len(gt_boxes))
        pred_multiset = Counter(it.label for it in receipt.items if it.status == "accept")
        gt_multiset = Counter(g["class"] for g in gts)
        correct = (pred_multiset == gt_multiset) and not receipt.needs_review
        wrong_accepted = pred_multiset - gt_multiset   # accepted items that are not on the tray
        missing = gt_multiset - pred_multiset
        e2e["trays"] += 1
        e2e["auto_correct"] += int(correct)
        e2e["flagged"] += int(receipt.needs_review)
        silent = (not receipt.needs_review) and (bool(wrong_accepted) or bool(missing))
        e2e["silent_error"] += int(silent)
        e2e["silent_overcharge"] += int(silent and bool(wrong_accepted))
        e2e["silent_undercharge"] += int(silent and bool(missing))
        if args.save_vis:
            from checkout.pipeline import draw_receipt
            cv2.imwrite(str(args.save_vis / Path(f).name), draw_receipt(img, receipt))

    gt_l = np.array([r[0] for r in rec_rows])
    top1 = np.array([r[1] for r in rec_rows])
    sims = np.array([r[2] for r in rec_rows])
    margins = np.array([r[3] for r in rec_rows])
    status = np.array([r[4] for r in rec_rows])
    seen = np.array([g in known for g in gt_l])
    correct1 = top1 == gt_l
    top3 = np.array([g in c[:3] for g, c in zip(gt_l, [r[5] for r in rec_rows])])
    acc_mask = status == "accept"
    conf = Counter((g, p) for g, p, s in zip(gt_l, top1, seen) if s and g != p)
    per_class = defaultdict(list)
    for g, c, s in zip(gt_l, correct1, seen):
        if s:
            per_class[g].append(c)
    worst = sorted(((k, float(np.mean(v)), len(v)) for k, v in per_class.items()), key=lambda t: t[1])[:10]

    report = {
        "split": args.split,
        "detector": args.detector,
        "detection": {
            "precision": det_tp / max(det_tp + det_fp, 1), "recall": det_tp / max(det_tp + det_fn, 1),
            "exact_count_acc": count_ok / max(len(by_file), 1),
        },
        "recognition_on_gt_crops": {
            "n": int(len(rec_rows)), "n_class_not_in_gallery": int((~seen).sum()),
            "top1": float(correct1[seen].mean()) if seen.any() else None,
            "top3": float(top3[seen].mean()) if seen.any() else None,
            "accept_rate": float(acc_mask.mean()),
            "precision_of_accepted": float(correct1[acc_mask].mean()) if acc_mask.any() else None,
            "uncertain_rate": float((status == "uncertain").mean()),
            "unknown_rate": float((status == "unknown").mean()),
            "top_confusions": [[g, p, n] for (g, p), n in conf.most_common(10)],
            "worst_classes": worst,
        },
        "end_to_end": {
            "trays": e2e["trays"],
            "auto_correct_rate": e2e["auto_correct"] / max(e2e["trays"], 1),
            "flagged_for_review_rate": e2e["flagged"] / max(e2e["trays"], 1),
            "silent_error_rate": e2e["silent_error"] / max(e2e["trays"], 1),
            "silent_overcharge_rate": e2e["silent_overcharge"] / max(e2e["trays"], 1),
            "silent_undercharge_rate": e2e["silent_undercharge"] / max(e2e["trays"], 1),
        },
        "thresholds": {"accept_sim": args.accept_sim, "margin": args.margin, "unknown_sim": args.unknown_sim},
    }
    if args.calibrate is not None:
        report["calibration"] = {
            "target_precision": args.calibrate,
            "suggested": calibrate(sims[seen], margins[seen], correct1[seen], args.calibrate),
            "note": "computed on this split's GT crops; calibrate on val, report on test",
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
