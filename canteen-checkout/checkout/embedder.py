"""Crop -> L2-normalised embedding.

TimmEmbedder   pretrained backbone from timm (default DINOv2 ViT-S/14), optionally
               with weights fine-tuned by finetune_embedder.py (ArcFace).
HistEmbedder   torch-free colour/texture histogram baseline; only meant for smoke
               tests and as a sanity floor for the learned embedder.
"""
from __future__ import annotations

import cv2
import numpy as np

from .crop import letterbox


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class HistEmbedder:
    name = "hist"

    def __init__(self, size: int = 128, bins=(12, 6, 6)):
        self.size, self.bins = size, bins

    def _one(self, crop: np.ndarray) -> np.ndarray:
        im = letterbox(crop, self.size)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        valid = (np.abs(im.astype(int) - 114).sum(-1) > 6).astype(np.uint8)  # ignore gray padding
        h = cv2.calcHist([hsv], [0, 1, 2], valid, list(self.bins), [0, 180, 0, 256, 0, 256]).ravel()
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        t = np.histogram(ang[valid > 0], bins=8, range=(0, 2 * np.pi), weights=mag[valid > 0])[0]
        shape = np.array([valid.mean(), crop.shape[0] / max(crop.shape[1], 1)])
        f = np.concatenate([np.sqrt(l2n(h)), 0.5 * l2n(t.astype(np.float64)), 0.3 * shape])
        return f.astype(np.float32)

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        return l2n(np.stack([self._one(c) for c in crops])).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(np.prod(self.bins)) + 8 + 2


class TimmEmbedder:
    name = "timm"

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", checkpoint: str | None = None,
                 device: str | None = None, img_size: int = 224, tta_flip: bool = True, batch_size: int = 32):
        import torch
        import timm

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.proj = None
        ckpt = None
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model_name = ckpt["model_name"]
            img_size = ckpt.get("img_size", img_size)
        self.model = create_backbone(model_name, img_size, pretrained=ckpt is None)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["backbone"])
            if ckpt.get("proj") is not None:
                w = ckpt["proj"]["weight"]
                self.proj = torch.nn.Linear(w.shape[1], w.shape[0], bias="bias" in ckpt["proj"])
                self.proj.load_state_dict(ckpt["proj"])
                self.proj.to(self.device).eval()
        self.model.to(self.device).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = np.array(cfg["mean"], np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg["std"], np.float32).reshape(1, 1, 3)
        self.img_size, self.tta_flip, self.batch_size = img_size, tta_flip, batch_size
        self.model_name = model_name

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        im = letterbox(crop, self.img_size)[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        return ((im - self.mean) / self.std).transpose(2, 0, 1)

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                x = torch.from_numpy(np.stack([self._prep(c) for c in crops[i:i + self.batch_size]])).to(self.device)
                f = self._forward(x)
                if self.tta_flip:
                    f = f + self._forward(torch.flip(x, dims=[3]))
                out.append(f.float().cpu().numpy())
        if not out:
            return np.zeros((0, 1), np.float32)
        return l2n(np.concatenate(out)).astype(np.float32)

    def _forward(self, x):
        f = self.model(x)
        if self.proj is not None:
            f = self.proj(f)
        return self.torch.nn.functional.normalize(f, dim=-1)


def create_backbone(model_name: str, img_size: int, pretrained: bool = True):
    import timm
    try:  # ViTs accept a custom input size; CNNs do not take the kwarg
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
    except TypeError:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


def build_embedder(cfg: dict):
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "timm")
    if kind == "hist":
        return HistEmbedder()
    if kind == "timm":
        return TimmEmbedder(**cfg)
    raise ValueError(f"unknown embedder type {kind!r}")
