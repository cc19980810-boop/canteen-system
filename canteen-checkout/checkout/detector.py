"""Detector wrappers.

YoloDetector wraps an Ultralytics YOLOv8/YOLO11 (detect or seg) model and can
run flip test-time augmentation fused with WBF, the same trick
lannguyen0910/food-recognition uses to squeeze extra recall out of YOLOv5.

OracleDetector returns ground-truth instances; it is used to evaluate the
recognizer in isolation and in the torch-free smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .boxes import iou_matrix, weighted_boxes_fusion


@dataclass
class Detection:
    box: np.ndarray                      # x1, y1, x2, y2 (pixels)
    score: float
    polygon: np.ndarray | None = None    # Nx2 pixels (seg models only)
    meta: dict = field(default_factory=dict)


class YoloDetector:
    def __init__(self, weights: str, conf: float = 0.3, iou: float = 0.6, imgsz: int = 640,
                 device: str | None = None, tta: bool = False, max_det: int = 50,
                 wbf_iou: float = 0.55):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device, self.tta, self.max_det, self.wbf_iou = device, tta, max_det, wbf_iou

    def _run(self, img):
        r = self.model.predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                               device=self.device, max_det=self.max_det, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        polys = list(r.masks.xy) if getattr(r, "masks", None) is not None else [None] * len(boxes)
        return boxes, scores, polys

    def __call__(self, img: np.ndarray) -> list[Detection]:
        h, w = img.shape[:2]
        b0, s0, p0 = self._run(img)
        if not self.tta:
            return [Detection(b, float(s), p if p is not None and len(p) >= 3 else None)
                    for b, s, p in zip(b0, s0, p0)]

        # horizontal-flip view, mapped back to original coordinates
        b1, s1, p1 = self._run(np.ascontiguousarray(img[:, ::-1]))
        if len(b1):
            b1 = b1.copy()
            b1[:, [0, 2]] = w - b1[:, [2, 0]]
        p1 = [None if p is None else np.stack([w - p[:, 0], p[:, 1]], 1) for p in p1]

        scale = np.array([w, h, w, h], dtype=np.float64)
        fb, fs, _ = weighted_boxes_fusion(
            [np.clip(b0 / scale, 0, 1).tolist(), np.clip(b1 / scale, 0, 1).tolist()],
            [s0.tolist(), s1.tolist()], [[0] * len(b0), [0] * len(b1)],
            iou_thr=self.wbf_iou, skip_box_thr=self.conf * 0.5)
        fb = np.asarray(fb).reshape(-1, 4) * scale
        # attach the outline of the best-overlapping single-view detection
        all_b = np.concatenate([b0, b1]) if len(b0) + len(b1) else np.zeros((0, 4))
        all_p = list(p0) + list(p1)
        ious = iou_matrix(fb, all_b)
        dets = []
        for i, (b, s) in enumerate(zip(fb, fs)):
            if s < self.conf:
                continue
            poly = all_p[int(np.argmax(ious[i]))] if ious.shape[1] else None
            dets.append(Detection(b, float(s), poly if poly is not None and len(poly) >= 3 else None))
        return dets


class OracleDetector:
    """Returns the ground-truth instances of an image (for recognizer-only evaluation)."""

    def __init__(self, instances_by_file: dict[str, list[dict]]):
        self.by_file = instances_by_file
        self.current: str | None = None

    def set_image(self, key: str):
        self.current = key

    def __call__(self, img: np.ndarray) -> list[Detection]:
        return [Detection(np.array(i["box"], float), 1.0, np.array(i["polygon"], float))
                for i in self.by_file.get(self.current, [])]
