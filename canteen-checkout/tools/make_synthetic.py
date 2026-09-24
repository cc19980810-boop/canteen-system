#!/usr/bin/env python3
"""Generate a small synthetic UNIMIB2016-like dataset for smoke tests.

Tray photos with 1-5 textured "dishes" of N classes, saved in the same folder
layout as the real release (images/, annotations.mat, split/*.mat). Two
classes are deliberately near-identical so the "uncertain" branch of the
recognizer gets exercised. This is only for checking that the code runs; it
says nothing about accuracy on real food.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

CLASSES = [
    # name,               BGR colour,       shape,     texture
    ("pasta_al_pomodoro", (40, 60, 200),   "ellipse", "stripes"),
    ("insalata_mista",    (60, 170, 70),   "blob",    "dots"),
    ("patate_fritte",     (60, 200, 230),  "rect",    "stripes"),
    ("pane",              (90, 140, 190),  "ellipse", "plain"),
    ("mandarini",         (30, 140, 245),  "circle",  "dots"),
    ("yogurt",            (235, 235, 240), "circle",  "plain"),
    ("budino",            (40, 80, 120),   "circle",  "plain"),
    ("budino_caramello",  (45, 85, 125),   "circle",  "plain"),   # near-duplicate of budino
]


def draw_item(img, rng, cls_idx):
    h, w = img.shape[:2]
    name, color, shape, tex = CLASSES[cls_idx]
    cx, cy = rng.integers(60, w - 60), rng.integers(60, h - 60)
    s = rng.integers(30, 60)
    mask = np.zeros((h, w), np.uint8)
    if shape == "circle":
        cv2.circle(mask, (int(cx), int(cy)), int(s), 255, -1)
    elif shape == "ellipse":
        cv2.ellipse(mask, (int(cx), int(cy)), (int(s * 1.3), int(s * 0.8)), float(rng.integers(0, 180)), 0, 360, 255, -1)
    elif shape == "rect":
        box = cv2.boxPoints(((float(cx), float(cy)), (float(s * 1.6), float(s)), float(rng.integers(0, 90))))
        cv2.fillPoly(mask, [box.astype(np.int32)], 255)
    else:
        ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
        r = s * rng.uniform(0.7, 1.2, 9)
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    layer = np.zeros_like(img)
    layer[:] = np.clip(np.array(color) + rng.normal(0, 8, 3), 0, 255)
    if tex == "stripes":
        for y in range(0, h, 8):
            cv2.line(layer, (0, y), (w, y + 20), tuple(int(c * 0.7) for c in color), 2)
    elif tex == "dots":
        for _ in range(250):
            cv2.circle(layer, (int(rng.integers(0, w)), int(rng.integers(0, h))), 2,
                       tuple(int(255 - c) for c in color), -1)
    noise = rng.normal(0, 6, img.shape)
    layer = np.clip(layer + noise, 0, 255).astype(np.uint8)
    img[mask > 0] = layer[mask > 0]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return name, c, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "split").mkdir(exist_ok=True)

    rows, names = [], []
    for i in range(args.n):
        name = f"2016{i:04d}_1200{i % 60:02d}"
        img = np.full((360, 480, 3), (200, 205, 210), np.uint8)
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        occupied = np.zeros(img.shape[:2], np.uint8)
        items = []
        for _ in range(int(rng.integers(1, 6))):
            for _try in range(20):
                trial = img.copy()
                cname, poly, mask = draw_item(trial, rng, int(rng.integers(len(CLASSES))))
                if (occupied[mask > 0] > 0).mean() < 0.02:
                    img = trial
                    occupied |= mask
                    x1, y1 = poly.min(0)
                    x2, y2 = poly.max(0)
                    items.append((cname, [x1, y1, x2 - x1, y2 - y1], poly.ravel()))  # BR = x1,y1,x2,y2,...
                    break
        cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), img)
        st = np.empty((len(items),), dtype=[("class", "O"), ("BoundingBox", "O"), ("BR", "O")])
        for k, (c, bb, br) in enumerate(items):
            st[k] = (c, np.array(bb, float), np.array(br, float))
        rows.append((name, st))
        names.append(name)

    cell = np.empty((len(rows), 2), dtype=object)
    for i, (n, st) in enumerate(rows):
        cell[i, 0], cell[i, 1] = n, st
    savemat(str(args.out / "annotations.mat"), {"annotations": cell})
    k = int(len(names) * 0.8)
    tr = np.empty((k, 1), dtype=object)
    te = np.empty((len(names) - k, 1), dtype=object)
    for i, n in enumerate(names[:k]):
        tr[i, 0] = n
    for i, n in enumerate(names[k:]):
        te[i, 0] = n
    savemat(str(args.out / "split" / "TrainingSet.mat"), {"TrainingSet": tr})
    savemat(str(args.out / "split" / "TestSet.mat"), {"TestSet": te})
    print(f"wrote {len(names)} synthetic trays to {args.out}")


if __name__ == "__main__":
    main()
