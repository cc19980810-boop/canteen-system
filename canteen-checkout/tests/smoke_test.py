#!/usr/bin/env python3
"""Torch-free smoke test of the whole pipeline on synthetic trays.

    python tests/smoke_test.py            (~20 s on a laptop CPU)

Covers: dataset conversion, gallery build/append/save/load, kNN decisions,
WBF, the YoloDetector flip-TTA logic (with a stubbed ultralytics.YOLO), the
checkout pipeline incl. price list and weight cross-check, and evaluate.py.
Uses the histogram embedder, so the accuracy numbers are meaningless.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkout.boxes import _wbf_numpy, greedy_match, iou_matrix  # noqa: E402
from checkout.detector import Detection, OracleDetector  # noqa: E402
from checkout.embedder import HistEmbedder  # noqa: E402
from checkout.index import VectorIndex  # noqa: E402
from checkout.pipeline import CheckoutPipeline, PriceList, draw_receipt  # noqa: E402


def run(*cmd):
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}")
    return r.stdout


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="canteen_smoke_"))
    print(f"workdir {tmp}")

    print("[1] synthetic data + conversion")
    run("tools/make_synthetic.py", "--out", tmp / "raw", "--n", 60)
    # like most real UNIMIB photos: raw pixels + an EXIF "rotate 90" tag, and a "(0)" download suffix
    from PIL import Image
    first = sorted((tmp / "raw" / "images").glob("*.jpg"))[0]
    raw = Image.open(first)
    exif = raw.getexif()
    exif[274] = 6
    exif_name = first.with_name(first.stem + "(0).jpg")
    raw.save(exif_name, exif=exif.tobytes())
    first.unlink()
    out = run("tools/convert_unimib.py", "--root", tmp / "raw", "--out", tmp / "ds")
    inst = json.loads((tmp / "ds" / "instances.json").read_text())
    check(len(inst) > 100, f"{len(inst)} instances parsed")
    lbl = next((tmp / "ds" / "labels" / "train").iterdir()).read_text().split("\n")[0].split()
    check(lbl[0] == "0" and len(lbl) >= 7 and all(0 <= float(v) <= 1 for v in lbl[1:]), "YOLO-seg label format")
    check((tmp / "ds" / "data.yaml").exists(), "data.yaml written")
    ex = next(i for i in inst if i["image"] == first.stem)
    out_img = tmp / "ds" / ex["file"]
    loaded = cv2.imread(str(out_img))  # OpenCV applies EXIF by default: must still be the raw grid
    check(not out_img.is_symlink() and loaded.shape[:2] == (360, 480),
          "EXIF-rotated '(0)' image matched and rewritten unrotated")

    print("[2] boxes")
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    check(abs(iou_matrix(a, a)[0, 0] - 1) < 1e-9 and iou_matrix(a, a)[0, 1] == 0, "IoU")
    check(len(greedy_match(a, a[::-1])) == 2, "greedy match")
    fb, fs, _ = _wbf_numpy([[[0.1, 0.1, 0.3, 0.3]], [[0.12, 0.1, 0.32, 0.3]]], [[0.9], [0.7]], [[0], [0]],
                           None, 0.55, 0.0)
    check(len(fb) == 1 and 0.1 < fb[0][0] < 0.12 and abs(fs[0] - 0.8) < 1e-6, "WBF fuses two views")

    print("[3] gallery")
    run("build_gallery.py", "--crops", tmp / "ds" / "crops" / "train", "--out", tmp / "g.npz", "--embedder", "hist")
    idx = VectorIndex.load(tmp / "g.npz")
    check(len(idx.class_names) == 8, "8 classes in gallery")
    emb = HistEmbedder()
    # new dish registered from photos only
    new = tmp / "new" / "torta_nuova"
    new.mkdir(parents=True)
    for i in range(3):
        im = np.full((100, 100, 3), 114, np.uint8)
        cv2.circle(im, (50, 50), 40, (200, 50, 200), -1)
        cv2.imwrite(str(new / f"{i}.jpg"), im)
    run("build_gallery.py", "--crops", tmp / "new", "--append", tmp / "g.npz", "--out", tmp / "g2.npz",
        "--embedder", "hist")
    idx2 = VectorIndex.load(tmp / "g2.npz")
    q = emb([cv2.imread(str(new / "0.jpg"))])
    d = idx2.classify(q, accept_sim=0.9, margin=0.0)[0]
    check(d.label == "torta_nuova" and d.status == "accept", "appended dish recognised without retraining")
    idx2.remove_class("torta_nuova")
    check("torta_nuova" not in [idx2.class_names[i] for i in set(idx2.labels.tolist())], "remove_class")
    # a class with many samples must not hide the runner-up (search depth expansion)
    big = VectorIndex(np.vstack([np.tile([[1, 0]], (100, 1)), [[0.8, 0.6]]]).astype(np.float32),
                      np.array([0] * 100 + [1]), ["a", "b"])
    r = big.classify(np.array([[1, 0]], np.float32), k=10, accept_sim=0.5, margin=0.1)[0]
    check(r.status == "accept" and abs(r.margin - 0.2) < 1e-5, "margin uses true runner-up")
    r = idx.classify(np.zeros((1, idx.vectors.shape[1]), np.float32) + 1e-3, unknown_sim=0.99)[0]
    check(r.status == "unknown" and r.label is None, "open-set rejection")

    print("[4] YoloDetector flip-TTA with a stubbed ultralytics")
    class _T:  # mimics a torch tensor's .cpu().numpy()
        def __init__(self, a): self.a = np.asarray(a, np.float32)
        def cpu(self): return self
        def numpy(self): return self.a

    class _YOLO:
        def __init__(self, w): pass
        def predict(self, img, **kw):
            h, w = img.shape[:2]
            flipped = img[0, 0, 0] == 7  # marker pixel tells which view we got
            box = [w - 60, 10, w - 20, 50] if flipped else [20, 10, 60, 50]
            poly = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]]], np.float32)
            r = types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=_T([box]), conf=_T([0.8])),
                                      masks=types.SimpleNamespace(xy=[poly]))
            return [r]
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=_YOLO)
    from checkout.detector import YoloDetector
    det = YoloDetector("stub.pt", tta=True)
    img = np.zeros((100, 200, 3), np.uint8)
    img[0, -1, 0] = 7  # after horizontal flip this lands on [0, 0]
    dets = det(img)
    check(len(dets) == 1 and np.allclose(dets[0].box, [20, 10, 60, 50], atol=1e-3),
          f"flipped view mapped back and fused: {np.round(dets[0].box, 2).tolist()}")
    check(dets[0].polygon is not None and dets[0].polygon[:, 0].max() <= 60 + 1e-3, "polygon mapped back")
    del sys.modules["ultralytics"]

    print("[5] pipeline: prices, jittered detector, false positive, weight check")
    prices = tmp / "prices.csv"
    with open(prices, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "price", "weight_g", "display_name"])
        for i, c in enumerate(idx.class_names):
            w.writerow([c, f"{1.5 + i * 0.5:.2f}", 100 + 20 * i, c.replace("_", " ").title()])
    test = [i for i in inst if i["split"] == "test"]
    by_file = {}
    for i in test:
        by_file.setdefault(i["file"], []).append(i)
    f0, gts = next(iter(by_file.items()))
    img = cv2.imread(str(tmp / "ds" / f0))
    rng = np.random.default_rng(0)

    class Jitter(OracleDetector):
        def __call__(self, im):
            ds = super().__call__(im)
            for d in ds:
                d.box = d.box + rng.normal(0, 2, 4)
            ds.append(Detection(np.array([0, 0, 25, 25], float), 0.4))  # empty-tray false positive
            return ds
    jd = Jitter(by_file)
    jd.set_image(f0)
    pl = PriceList(prices)
    pipe = CheckoutPipeline(jd, emb, idx, pl, accept_sim=0.85, margin=0.0, unknown_sim=0.8)
    exp_w = sum(pl.weight(g["class"]) for g in gts)
    rec = pipe(img, measured_weight_g=exp_w)
    check(len(rec.items) == len(gts) + 1, f"{len(rec.items)} line items for {len(gts)} dishes + 1 false positive")
    check(rec.needs_review, "false positive forces review: " + "; ".join(rec.review_reasons))
    jd2 = OracleDetector(by_file)
    jd2.set_image(f0)
    pipe2 = CheckoutPipeline(jd2, emb, idx, pl, accept_sim=0.5, margin=0.0)
    ok = pipe2(img, measured_weight_g=exp_w)
    bad = pipe2(img, measured_weight_g=exp_w * 1.6)
    check(ok.expected_weight_g is not None and not any("weight" in r for r in ok.review_reasons),
          f"weight consistent: expected {ok.expected_weight_g} g, total {ok.total}")
    check(any("weight mismatch" in r for r in bad.review_reasons), "weight mismatch flagged")
    cv2.imwrite(str(tmp / "receipt.jpg"), draw_receipt(img, rec))

    print("[6] evaluate.py")
    out = run("evaluate.py", "--dataset", tmp / "ds", "--gallery", tmp / "g.npz", "--split", "val",
              "--calibrate", "0.95", "--out", tmp / "report.json")
    rep = json.loads((tmp / "report.json").read_text())
    check(rep["detection"]["recall"] == 1.0, "oracle detector recall = 1")
    check(rep["recognition_on_gt_crops"]["top3"] is not None, f"recognition top1 = "
          f"{rep['recognition_on_gt_crops']['top1']:.2f} (hist baseline)")
    check("silent_error_rate" in rep["end_to_end"], "end-to-end silent error rate reported")
    print(f"\nall smoke tests passed; artefacts in {tmp}")


if __name__ == "__main__":
    main()
