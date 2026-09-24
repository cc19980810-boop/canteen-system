#!/usr/bin/env python3
"""Rename the classes of an existing gallery.npz with a class map, without re-embedding.

Class names are only labels on the gallery rows, so a gallery built from the Italian
UNIMIB crops can be switched to English names in place of a rebuild:

    python tools/relabel_gallery.py --gallery gallery.npz --out gallery_en.npz

Classes missing from the map keep their name. Crop paths stored in `sources` are
rewritten too (crops/<split>/<old>/x.jpg -> crops/<split>/<new>/x.jpg).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkout.index import VectorIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--class-map", type=Path, default=Path(__file__).resolve().parent / "class_names_en.csv",
                    help="CSV with columns italian,english (default: tools/class_names_en.csv)")
    args = ap.parse_args()

    with open(args.class_map, newline="") as f:
        cmap = {r["italian"]: r["english"] for r in csv.DictReader(f)}
    index = VectorIndex.load(args.gallery)
    new_names = [cmap.get(c, c) for c in index.class_names]
    if len(set(new_names)) != len(new_names):
        sys.exit("renaming would merge two classes (gallery already mixes old and new names?)")
    renamed = sum(a != b for a, b in zip(index.class_names, new_names))

    sources = []
    for s in index.sources:
        parts = Path(s).parts
        if len(parts) >= 2 and parts[-2] in cmap:
            s = str(Path(*parts[:-2], cmap[parts[-2]], parts[-1]))
        sources.append(s)

    out = VectorIndex(index.vectors, index.labels, new_names, sources, index.meta)
    out.save(args.out)
    print(f"renamed {renamed}/{len(new_names)} classes; unchanged: "
          f"{[c for c in new_names if c not in cmap.values()]}")
    print(out.summary())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
