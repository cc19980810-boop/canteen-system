"""Box utilities: IoU, greedy matching and Weighted Boxes Fusion.

WBF follows the idea used in lannguyen0910/food-recognition (ensemble_boxes):
boxes predicted by several models / TTA views are clustered and averaged
instead of NMS-suppressed. If the `ensemble-boxes` package is installed it is
used; otherwise a compact reimplementation below is used.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5):
    """Greedy one-to-one matching by IoU. Returns list of (pred_idx, gt_idx, iou)."""
    iou = iou_matrix(pred, gt)
    pairs = []
    if iou.size == 0:
        return pairs
    used_p, used_g = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for p, g in order:
        if iou[p, g] < thr:
            break
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        pairs.append((int(p), int(g), float(iou[p, g])))
    return pairs


def _wbf_numpy(boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr):
    n_models = len(boxes_list)
    weights = np.ones(n_models) if weights is None else np.asarray(weights, dtype=np.float64)
    entries = []  # label, score*w, x1, y1, x2, y2
    for m in range(n_models):
        for b, s, l in zip(boxes_list[m], scores_list[m], labels_list[m]):
            if s < skip_box_thr:
                continue
            entries.append((l, s * weights[m], *b))
    if not entries:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    entries = np.array(entries, dtype=np.float64)
    out_b, out_s, out_l = [], [], []
    for lab in np.unique(entries[:, 0]):
        e = entries[entries[:, 0] == lab]
        e = e[np.argsort(-e[:, 1])]
        clusters, fused = [], []
        for row in e:
            best, best_iou = -1, iou_thr
            if fused:
                ious = iou_matrix(row[None, 2:6], np.array(fused))[0]
                j = int(np.argmax(ious))
                if ious[j] > best_iou:
                    best = j
            if best < 0:
                clusters.append([row])
                fused.append(row[2:6].copy())
            else:
                clusters[best].append(row)
                c = np.array(clusters[best])
                fused[best] = (c[:, 2:6] * c[:, 1:2]).sum(0) / c[:, 1].sum()
        for c, fb in zip(clusters, fused):
            c = np.array(c)
            score = c[:, 1].mean() * min(len(c), n_models) / weights.sum()
            out_b.append(fb)
            out_s.append(score)
            out_l.append(lab)
    order = np.argsort(-np.array(out_s))
    return np.array(out_b)[order], np.array(out_s)[order], np.array(out_l)[order]


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None,
                          iou_thr=0.55, skip_box_thr=0.0):
    """Boxes must be normalised to [0, 1] (x1, y1, x2, y2)."""
    try:
        from ensemble_boxes import weighted_boxes_fusion as _wbf  # type: ignore
        return _wbf(boxes_list, scores_list, labels_list, weights=weights,
                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    except ImportError:
        return _wbf_numpy(boxes_list, scores_list, labels_list, weights, iou_thr, skip_box_thr)
