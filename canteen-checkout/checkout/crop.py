"""Instance cropping shared by dataset conversion, gallery building and inference.

Using the *same* function everywhere matters: if gallery crops are masked and
tightly padded but runtime crops are not, the embeddings drift apart and
retrieval accuracy drops for no visible reason.
"""
from __future__ import annotations

import cv2
import numpy as np

GRAY = (114, 114, 114)


def polygon_to_mask(polygon: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(np.asarray(polygon, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def crop_instance(
    img: np.ndarray,
    box,
    polygon: np.ndarray | None = None,
    pad: float = 0.08,
    mask_bg: bool = True,
    fill=GRAY,
) -> np.ndarray:
    """Crop one food item.

    img:      HxWx3 BGR image
    box:      (x1, y1, x2, y2) in pixels
    polygon:  optional Nx2 instance outline in pixels; used to blank out the
              tray / neighbouring dishes when mask_bg is True
    pad:      relative padding added around the box
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, np.floor(x1 - pad * bw)))
    y1 = int(max(0, np.floor(y1 - pad * bh)))
    x2 = int(min(w, np.ceil(x2 + pad * bw)))
    y2 = int(min(h, np.ceil(y2 + pad * bh)))
    if x2 <= x1 or y2 <= y1:
        return np.full((8, 8, 3), fill, dtype=np.uint8)

    if polygon is not None and mask_bg and len(polygon) >= 3:
        sub = img[y1:y2, x1:x2].copy()
        local = np.asarray(polygon, dtype=np.float32) - np.array([x1, y1], dtype=np.float32)
        m = polygon_to_mask(local, y2 - y1, x2 - x1)
        # slight dilation so that the edge of the food is kept
        m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
        sub[m == 0] = fill
        return sub
    return img[y1:y2, x1:x2].copy()


def letterbox(img: np.ndarray, size: int, fill=GRAY) -> np.ndarray:
    """Resize keeping aspect ratio and pad to size x size (shape is informative for food)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out
