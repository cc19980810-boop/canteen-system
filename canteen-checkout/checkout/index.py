"""Vector gallery + kNN recognizer with open-set rejection.

Each gallery row is one reference crop of one dish class. At query time the
top-K neighbours are retrieved (FAISS inner product if installed, numpy
otherwise; vectors are L2-normalised so IP = cosine). A class score is the
best similarity among that class's neighbours. Classes not in the top-K are
bounded above by the K-th similarity, which keeps the margin conservative.

Decision:
    accept     top1 >= accept_sim  and  top1 - top2 >= margin
    uncertain  top1 >= unknown_sim (show top candidates, ask the customer/cashier)
    unknown    otherwise           (not on today's menu / not food / detector error)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Decision:
    status: str                 # accept | uncertain | unknown
    label: str | None
    score: float
    margin: float
    candidates: list = field(default_factory=list)   # [(label, score), ...]


class VectorIndex:
    def __init__(self, vectors: np.ndarray, labels: np.ndarray, class_names: list[str],
                 sources: list[str] | None = None, meta: dict | None = None):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.class_names = list(class_names)
        self.sources = list(sources) if sources is not None else [""] * len(labels)
        self.meta = dict(meta or {})
        self._faiss = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        try:
            import faiss  # type: ignore
            idx = faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(self.vectors)
            self._faiss = idx
        except ImportError:
            self._faiss = None

    def add(self, vectors: np.ndarray, class_name: str, sources: list[str] | None = None):
        """Register a new dish (or more photos of an existing one) without retraining."""
        if class_name not in self.class_names:
            self.class_names.append(class_name)
        cid = self.class_names.index(class_name)
        self.vectors = np.concatenate([self.vectors, vectors.astype(np.float32)])
        self.labels = np.concatenate([self.labels, np.full(len(vectors), cid)])
        self.sources += list(sources or [""] * len(vectors))
        self._build()

    def remove_class(self, class_name: str):
        cid = self.class_names.index(class_name)
        keep = self.labels != cid
        self.vectors, self.labels = self.vectors[keep], self.labels[keep]
        self.sources = [s for s, k in zip(self.sources, keep) if k]
        self._build()

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path):
        np.savez_compressed(path, vectors=self.vectors, labels=self.labels,
                            class_names=np.array(self.class_names, dtype=object),
                            sources=np.array(self.sources, dtype=object),
                            meta=np.array([self.meta], dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        d = np.load(path, allow_pickle=True)
        meta = d["meta"][0] if "meta" in d else {}
        return cls(d["vectors"], d["labels"], d["class_names"].tolist(), d["sources"].tolist(), meta)

    # ------------------------------------------------------------------ query
    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, self.vectors.shape[1])
        k = min(k, len(self.vectors))
        if self._faiss is not None:
            return self._faiss.search(q, k)
        sims = q @ self.vectors.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, 1), idx

    def class_scores(self, q: np.ndarray, k: int = 50):
        """Per query: (ranked [(label, score)], floor). Search depth grows until at
        least two classes are seen, so a well-populated class cannot fill the
        whole top-k and hide its runner-up."""
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, self.vectors.shape[1])
        results = []
        n = len(self.vectors)
        for row in q:
            kk = min(k, n)
            while True:
                sims, idx = self.search(row[None], kk)
                srow, irow = sims[0], idx[0]
                best: dict[int, float] = {}
                for s, i in zip(srow, irow):
                    if i < 0:
                        continue
                    c = int(self.labels[i])
                    if s > best.get(c, -2.0):
                        best[c] = float(s)
                if len(best) >= 2 or kk >= n:
                    break
                kk = min(n, kk * 4)
            ranked = sorted(best.items(), key=lambda t: -t[1])
            floor = float(srow[-1]) if kk < n else -1.0  # any unseen class scores <= floor
            results.append(([(self.class_names[c], s) for c, s in ranked], floor))
        return results

    def classify(self, q: np.ndarray, k: int = 50, accept_sim: float = 0.6, margin: float = 0.05,
                 unknown_sim: float = 0.35, n_candidates: int = 3) -> list[Decision]:
        out = []
        for ranked, floor in self.class_scores(q, k):
            top1_lbl, top1 = ranked[0]
            top2 = ranked[1][1] if len(ranked) > 1 else floor
            m = top1 - top2
            if top1 >= accept_sim and m >= margin:
                status = "accept"
            elif top1 >= unknown_sim:
                status = "uncertain"
            else:
                status = "unknown"
            out.append(Decision(status, top1_lbl if status != "unknown" else None, top1, m,
                                ranked[:n_candidates]))
        return out

    def __len__(self):
        return len(self.labels)

    def summary(self) -> str:
        counts = np.bincount(self.labels, minlength=len(self.class_names))
        return (f"{len(self)} vectors, {len(self.class_names)} classes, dim={self.vectors.shape[1]}, "
                f"per-class min/median/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}, "
                f"backend={'faiss' if self._faiss is not None else 'numpy'}")
