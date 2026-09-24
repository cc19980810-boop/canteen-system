#!/usr/bin/env python3
"""Fine-tune the retrieval backbone on dish crops with an ArcFace head.

Off-the-shelf DINOv2 features already work reasonably for retrieval; ArcFace
fine-tuning pulls crops of the same dish together and pushes look-alike dishes
(e.g. two kinds of pudding) apart. The classifier head is thrown away after
training: only the backbone (+ a small projection) is kept, so new dishes can
still be added to the gallery without retraining.

    python finetune_embedder.py --train data/unimib_yolo/crops/train \
        --val data/unimib_yolo/crops/val --out runs/embedder

Validation = retrieval top-1 of val crops against a gallery of train crops,
i.e. exactly how the model is used at the checkout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkout.crop import letterbox  # noqa: E402
from checkout.index import VectorIndex  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # keeps --help working without torch
    torch, Dataset = None, object


class Crops(Dataset):
    """Module-level so DataLoader workers can pickle it (spawn on macOS / Windows)."""

    def __init__(self, items, train, img_size, mean, std):
        self.items, self.train = items, train
        self.img_size, self.mean, self.std = img_size, mean, std

    def __len__(self):
        return len(self.items)

    def _aug(self, im):
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            im = im[:, ::-1]
        if rng.random() < 0.5:
            im = im[::-1]
        k = int(rng.integers(0, 4))
        im = np.rot90(im, k)
        h, w = im.shape[:2]  # random crop keeping 75-100 %
        s = rng.uniform(0.75, 1.0)
        ch, cw = max(8, int(h * s)), max(8, int(w * s))
        y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        im = np.ascontiguousarray(im[y:y + ch, x:x + cw])
        # brightness / contrast only: hue is a class cue for food
        im = np.clip(im.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
        return im.astype(np.uint8)

    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.imread(str(p))
        if self.train:
            im = self._aug(im)
        im = letterbox(im, self.img_size)[:, :, ::-1].astype(np.float32) / 255.0
        im = ((im - self.mean) / self.std).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(im)), y



def list_folder(root: Path, class_names: list[str] | None = None):
    names = class_names or sorted(d.name for d in root.iterdir() if d.is_dir())
    items = []
    for ci, n in enumerate(names):
        d = root / n
        if d.is_dir():
            items += [(p, ci) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]
    return names, items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--val", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/embedder"))
    ap.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--emb-dim", type=int, default=256, help="0 = no projection layer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="head / projection LR")
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--freeze-blocks", type=int, default=8,
                    help="ViT: freeze patch embed + first N blocks (small data!). -1 = freeze all")
    ap.add_argument("--arc-s", type=float, default=30.0)
    ap.add_argument("--arc-m", type=float, default=0.3)
    ap.add_argument("--balanced", action="store_true", help="class-balanced sampling (UNIMIB is long-tailed)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import timm
    from checkout.embedder import create_backbone

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    names, train_items = list_folder(args.train)
    _, val_items = list_folder(args.val, names) if args.val else (names, [])
    print(f"{len(names)} classes, {len(train_items)} train crops, {len(val_items)} val crops, device={device}")

    backbone = create_backbone(args.model_name, args.img_size, pretrained=True)
    cfg = timm.data.resolve_model_data_config(backbone)
    mean = np.array(cfg["mean"], np.float32).reshape(1, 1, 3)
    std = np.array(cfg["std"], np.float32).reshape(1, 1, 3)
    feat_dim = backbone.num_features

    class ArcFace(nn.Module):
        def __init__(self, dim, n, s, m):
            super().__init__()
            self.w = nn.Parameter(torch.empty(n, dim))
            nn.init.xavier_uniform_(self.w)
            self.s, self.m = s, m

        def forward(self, f, y):
            cos = F.linear(F.normalize(f), F.normalize(self.w)).clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            target = torch.cos(theta + self.m)
            onehot = F.one_hot(y, cos.shape[1]).bool()
            logits = torch.where(onehot, target, cos) * self.s
            return F.cross_entropy(logits, y)

    proj = nn.Linear(feat_dim, args.emb_dim) if args.emb_dim > 0 else None
    out_dim = args.emb_dim if args.emb_dim > 0 else feat_dim
    head = ArcFace(out_dim, len(names), args.arc_s, args.arc_m)

    # freezing: small datasets overfit a full ViT quickly
    if args.freeze_blocks != 0:
        for p in backbone.parameters():
            p.requires_grad = False
        if args.freeze_blocks > 0 and hasattr(backbone, "blocks"):
            for b in backbone.blocks[args.freeze_blocks:]:
                for p in b.parameters():
                    p.requires_grad = True
            for mod in ("norm", "fc_norm"):
                if getattr(backbone, mod, None) is not None:
                    for p in getattr(backbone, mod).parameters():
                        p.requires_grad = True
        elif args.freeze_blocks > 0:
            print("backbone has no .blocks (not a ViT); training all layers at backbone-lr")
            for p in backbone.parameters():
                p.requires_grad = True

    backbone.to(device)
    head.to(device)
    if proj is not None:
        proj.to(device)
    groups = [{"params": list(head.parameters()), "lr": args.lr}]
    if proj is not None:
        groups.append({"params": list(proj.parameters()), "lr": args.lr})
    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)

    train_ds = Crops(train_items, True, args.img_size, mean, std)
    if args.balanced:
        counts = np.bincount([y for _, y in train_items], minlength=len(names))
        weights = [1.0 / counts[y] for _, y in train_items]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
        loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True)
    else:
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    total_steps = max(1, args.epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, len(loader))) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))

    def embed(items):
        backbone.eval()
        if proj is not None:
            proj.eval()
        dl = DataLoader(Crops(items, False, args.img_size, mean, std), batch_size=args.batch, num_workers=args.workers)
        vecs = []
        with torch.inference_mode():
            for x, _ in dl:
                f = backbone(x.to(device))
                if proj is not None:
                    f = proj(f)
                vecs.append(F.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(vecs)

    def evaluate():
        if not val_items:
            return None
        g = embed(train_items)
        q = embed(val_items)
        idx = VectorIndex(g, np.array([y for _, y in train_items]), names)
        pred = [r[0][0][0] for r in idx.class_scores(q, k=20)]
        return float(np.mean([p == names[y] for p, (_, y) in zip(pred, val_items)]))

    def save(path):
        torch.save({"model_name": args.model_name, "img_size": args.img_size,
                    "backbone": backbone.state_dict(),
                    "proj": proj.state_dict() if proj is not None else None,
                    "class_names": names}, path)

    best = evaluate()
    print(f"epoch 0 (pretrained) val retrieval top1 = {best}")
    best = best if best is not None else -1.0
    history = []
    for ep in range(1, args.epochs + 1):
        backbone.train()
        if proj is not None:
            proj.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            f = backbone(x)
            if proj is not None:
                f = proj(f)
            loss = head(f, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        acc = evaluate()
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_top1": acc})
        print(f"epoch {ep}: loss={np.mean(losses):.3f} val_top1={acc}")
        save(args.out / "last.pt")
        if acc is None or acc > best:
            best = acc if acc is not None else best
            save(args.out / "best.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val top1 = {best}; checkpoint: {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
