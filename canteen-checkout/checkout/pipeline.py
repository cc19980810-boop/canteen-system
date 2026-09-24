"""End-to-end checkout: detect -> crop -> embed -> retrieve -> decide -> price -> cross-check."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .crop import crop_instance
from .detector import Detection
from .index import Decision, VectorIndex


@dataclass
class LineItem:
    box: list
    det_score: float
    status: str                 # accept | uncertain | unknown
    label: str | None
    sim: float
    margin: float
    candidates: list
    price: float | None = None
    display_name: str | None = None


@dataclass
class Receipt:
    items: list = field(default_factory=list)
    total: float = 0.0
    needs_review: bool = False
    review_reasons: list = field(default_factory=list)
    expected_weight_g: float | None = None
    measured_weight_g: float | None = None

    def to_dict(self):
        return asdict(self)


class PriceList:
    """CSV with columns: class, price, weight_g (optional), display_name (optional)."""

    def __init__(self, path: str | Path | None):
        self.rows: dict[str, dict] = {}
        if path and Path(path).exists():
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    self.rows[r["class"]] = r

    def price(self, label):
        v = self.rows.get(label, {}).get("price", "")
        return float(v) if v not in ("", None) else None

    def weight(self, label):
        v = self.rows.get(label, {}).get("weight_g", "")
        return float(v) if v not in ("", None) else None

    def display(self, label):
        return self.rows.get(label, {}).get("display_name") or label


class CheckoutPipeline:
    def __init__(self, detector, embedder, index: VectorIndex, prices: PriceList | None = None,
                 k: int = 50, accept_sim: float = 0.6, margin: float = 0.05, unknown_sim: float = 0.35,
                 crop_pad: float = 0.08, mask_bg: bool = True, weight_tolerance: float = 0.15,
                 min_box_frac: float = 0.002):
        self.detector, self.embedder, self.index = detector, embedder, index
        self.prices = prices or PriceList(None)
        self.k, self.accept_sim, self.margin, self.unknown_sim = k, accept_sim, margin, unknown_sim
        self.crop_pad, self.mask_bg = crop_pad, mask_bg
        self.weight_tolerance, self.min_box_frac = weight_tolerance, min_box_frac

    @classmethod
    def from_config(cls, path: str | Path, **overrides):
        from .embedder import build_embedder
        from .detector import YoloDetector
        cfg = yaml.safe_load(Path(path).read_text())
        cfg_dir = Path(path).resolve().parent

        def rel(p):
            return None if p in (None, "") else str((cfg_dir / p).resolve()) if not Path(p).is_absolute() else p

        det_cfg = dict(cfg["detector"])
        det_cfg["weights"] = rel(det_cfg["weights"])
        emb_cfg = dict(cfg.get("embedder", {}))
        if emb_cfg.get("checkpoint"):
            emb_cfg["checkpoint"] = rel(emb_cfg["checkpoint"])
        index = VectorIndex.load(rel(cfg["index"]["path"]))
        dec, crop, pr = cfg.get("decision", {}), cfg.get("crop", {}), cfg.get("pricing", {})
        kw = dict(k=cfg["index"].get("k", 50), accept_sim=dec.get("accept_sim", 0.6),
                  margin=dec.get("margin", 0.05), unknown_sim=dec.get("unknown_sim", 0.35),
                  crop_pad=crop.get("pad", 0.08), mask_bg=crop.get("mask_bg", True),
                  weight_tolerance=pr.get("weight_tolerance", 0.15))
        kw.update(overrides)
        return cls(YoloDetector(**det_cfg), build_embedder(emb_cfg), index,
                   PriceList(rel(pr.get("csv"))), **kw)

    # ------------------------------------------------------------------
    def recognize(self, img: np.ndarray, dets: list[Detection]) -> list[Decision]:
        if not dets:
            return []
        crops = [crop_instance(img, d.box, d.polygon, pad=self.crop_pad, mask_bg=self.mask_bg) for d in dets]
        q = self.embedder(crops)
        return self.index.classify(q, k=self.k, accept_sim=self.accept_sim, margin=self.margin,
                                   unknown_sim=self.unknown_sim)

    def __call__(self, img: np.ndarray, measured_weight_g: float | None = None) -> Receipt:
        h, w = img.shape[:2]
        dets = [d for d in self.detector(img)
                if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) >= self.min_box_frac * h * w]
        decisions = self.recognize(img, dets)
        rec = Receipt(measured_weight_g=measured_weight_g)
        exp_w, have_all_w = 0.0, True
        for d, dec in zip(dets, decisions):
            price = self.prices.price(dec.label) if dec.status == "accept" else None
            item = LineItem(box=[round(float(v), 1) for v in d.box], det_score=round(d.score, 3),
                            status=dec.status, label=dec.label, sim=round(dec.score, 4),
                            margin=round(dec.margin, 4),
                            candidates=[(c, round(s, 4)) for c, s in dec.candidates],
                            price=price, display_name=self.prices.display(dec.label) if dec.label else None)
            rec.items.append(item)
            if dec.status == "accept":
                if price is None:
                    if self.prices.rows:  # only when a price list is configured
                        rec.review_reasons.append(f"no price for '{dec.label}'")
                else:
                    rec.total += price
                wgt = self.prices.weight(dec.label)
                if wgt is None:
                    have_all_w = False
                else:
                    exp_w += wgt
            else:
                have_all_w = False
        n_unc = sum(i.status == "uncertain" for i in rec.items)
        n_unk = sum(i.status == "unknown" for i in rec.items)
        if n_unc:
            rec.review_reasons.append(f"{n_unc} item(s) need confirmation")
        if n_unk:
            rec.review_reasons.append(f"{n_unk} unrecognised item(s)")
        if not rec.items:
            rec.review_reasons.append("no food detected")
        # Independent sensor cross-check: vision says X items weighing ~E g, scale says M g.
        if have_all_w and rec.items:
            rec.expected_weight_g = round(exp_w, 1)
            if measured_weight_g is not None and exp_w > 0:
                rel_err = abs(measured_weight_g - exp_w) / exp_w
                if rel_err > self.weight_tolerance:
                    rec.review_reasons.append(
                        f"weight mismatch: expected ~{exp_w:.0f} g, measured {measured_weight_g:.0f} g")
        rec.total = round(rec.total, 2)
        rec.needs_review = bool(rec.review_reasons)
        return rec


COLORS = {"accept": (60, 180, 60), "uncertain": (0, 170, 255), "unknown": (40, 40, 220)}


def draw_receipt(img: np.ndarray, rec: Receipt) -> np.ndarray:
    out = img.copy()
    for i, it in enumerate(rec.items):
        x1, y1, x2, y2 = map(int, it.box)
        c = COLORS[it.status]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        if it.status == "accept":
            txt = f"{i + 1}. {it.display_name} {it.sim:.2f}"
        elif it.status == "uncertain":
            txt = f"{i + 1}. ? " + " / ".join(n for n, _ in it.candidates[:2])
        else:
            txt = f"{i + 1}. unknown"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx = int(min(max(0, x1), max(0, out.shape[1] - tw - 4)))       # keep label inside the image
        ly = y1 if y1 - th - 6 >= 0 else min(out.shape[0] - 1, y1 + th + 6)
        cv2.rectangle(out, (lx, ly - th - 6), (lx + tw + 4, ly), c, -1)
        cv2.putText(out, txt, (lx + 2, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    footer = f"total {rec.total:.2f}" + ("  REVIEW" if rec.needs_review else "")
    cv2.putText(out, footer, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, footer, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out
