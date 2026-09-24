#!/usr/bin/env python3
"""Convert UNIMIB2016 into (a) a YOLO dataset for a class-agnostic food detector
and (b) per-class instance crops for the retrieval gallery.

Expected input (the official release from the IVL lab, Univ. Milano-Bicocca):

    UNIMIB2016/
      images/            *.jpg tray photos
      annotations.mat    per-image food items: class name + BR (boundary polygon)
      split/TrainingSet.mat, split/TestSet.mat   (optional official split)

The exact nesting of annotations.mat is not documented in one place and third
party re-exports differ, so the parser does NOT hard-code a layout. It walks the
whole MATLAB structure and collects every node that has a `BR` field, taking
  * the image name from the nearest enclosing key / sibling string that matches
    a file in images/, and
  * the class name from a `class`/`name` field, else from the enclosing key.
A JSON re-export such as gist-ailab/Food-Instance-Segmentation's
annotations/{train,test}.json (image -> list of flat polygons, no class names)
is also accepted; items then get the single class "food".

Run with --inspect first if the instance count looks wrong: it prints the
structure of the file so the parser can be adjusted.

Output:
    OUT/images/{train,val,test}/*.jpg     (symlinks by default)
    OUT/labels/{train,val,test}/*.txt     YOLO-seg (default) or YOLO-detect labels, class 0 = food
    OUT/data.yaml                         for ultralytics
    OUT/crops/{train,val,test}/<class>/*.jpg
    OUT/instances.json                    every instance with split, class, box, polygon
    OUT/classes.txt, OUT/prices_template.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkout.crop import crop_instance  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------
# annotation loading
# ----------------------------------------------------------------------------
def load_any(path: Path):
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    from scipy.io import loadmat
    d = loadmat(str(path), simplify_cells=True)
    if "__function_workspace__" in d:          # MATLAB object (the official release uses containers.Map)
        m = read_containers_map(path)
        if m is not None:
            return m
    return d


def read_containers_map(path: Path):
    """Decode a MATLAB containers.Map saved with `save`, as in the official UNIMIB2016
    annotations.mat. scipy only exposes such objects as an opaque
    __function_workspace__ blob; that blob is itself a MAT-v5 stream whose
    FileWrapper__ object holds the Map's `keys` and `values` properties.

    Returns {image_name: [{"category", "class", "name", "boundary_type", "BR", "BoundingBox"}, ...]}.
    Tuple layout per the dataset's demo.m: (1) category (2) class (3) item name
    (4) boundary type (5) boundary points [x1,y1,...] (6) bounding box [x1,y1,...,x4,y4].
    """
    import io
    from scipy.io import loadmat
    from scipy.io.matlab._mio5 import MatFile5Reader

    blob = loadmat(str(path))["__function_workspace__"].tobytes()
    header = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + b"\x00\x01" + blob[2:4]
    reader = MatFile5Reader(io.BytesIO(header + blob[8:]), struct_as_record=True, squeeze_me=True,
                            chars_as_strings=True)
    reader.initialize_read()
    reader.mat_stream.seek(128)
    hdr, _ = reader.read_var_header()
    ws = reader.read_var_array(hdr, process=True)

    def find_map(o, depth=0):
        if depth > 8:
            return None
        if isinstance(o, np.ndarray) and o.dtype.names:
            if "keys" in o.dtype.names and "values" in o.dtype.names:
                return o
            for n in o.dtype.names:
                for c in np.ravel(o[n]):
                    r = find_map(c, depth + 1)
                    if r is not None:
                        return r
        elif isinstance(o, np.ndarray) and o.dtype == object:
            for c in o.ravel():
                r = find_map(c, depth + 1)
                if r is not None:
                    return r
        return None

    m = find_map(ws)
    if m is None:
        return None
    keys = np.ravel(m["keys"].item() if m.shape == () else m["keys"].ravel()[0])
    vals = np.ravel(m["values"].item() if m.shape == () else m["values"].ravel()[0])
    out = {}
    fields = ["category", "class", "name", "boundary_type", "BR", "BoundingBox"]
    for k, v in zip(keys, vals):
        rows = np.atleast_2d(np.asarray(v, dtype=object)) if not (isinstance(v, np.ndarray) and v.ndim == 2) else v
        items = []
        for row in rows:
            row = list(row)
            items.append({f: row[i] for i, f in enumerate(fields) if i < len(row)})
        out[str(k)] = items
    return {"annotations": out}


def inspect(obj, depth=0, max_depth=5, max_items=3, key="<root>"):
    pad = "  " * depth
    if hasattr(obj, "_fieldnames"):
        obj = {f: getattr(obj, f) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 2:
        print(f"{pad}{key}: cell{obj.shape}")
        if depth < max_depth:
            for i, row in enumerate(obj[:max_items]):
                inspect(row.tolist(), depth + 1, max_depth, max_items, f"row[{i}]")
        return
    if isinstance(obj, dict):
        print(f"{pad}{key}: dict[{len(obj)}] keys={list(obj)[:8]}")
        if depth < max_depth:
            for k in [k for k in obj if not str(k).startswith("__")][:max_items]:
                inspect(obj[k], depth + 1, max_depth, max_items, k)
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        seq = list(obj) if not isinstance(obj, np.ndarray) else obj.ravel().tolist()
        print(f"{pad}{key}: list[{len(seq)}]")
        if depth < max_depth:
            for i, c in enumerate(seq[:max_items]):
                inspect(c, depth + 1, max_depth, max_items, f"[{i}]")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{key}: ndarray{obj.shape} {obj.dtype} {obj.ravel()[:6]}")
    else:
        print(f"{pad}{key}: {type(obj).__name__} = {str(obj)[:60]!r}")


def _is_number_seq(x) -> bool:
    if isinstance(x, np.ndarray):
        return x.dtype != object and x.size >= 6
    if isinstance(x, (list, tuple)) and len(x) >= 6:
        return all(isinstance(v, (int, float, np.integer, np.floating)) for v in x)
    return False


class Walker:
    def __init__(self, image_stems: set[str]):
        self.stems = image_stems
        self.found: list[tuple[str, str | None, object]] = []

    def match_image(self, s) -> str | None:
        if not isinstance(s, str):
            return None
        s = s.strip()
        for cand in (s, Path(s).stem, s.lstrip("xX_"), Path(s.lstrip("xX_")).stem,
                     re.sub(r"^(img|image)_?", "", s, flags=re.I)):
            if cand in self.stems:
                return cand
        return None

    def walk(self, obj, img=None, cls_hint=None):
        if hasattr(obj, "_fieldnames"):  # scipy mat_struct left inside object arrays
            obj = {f: getattr(obj, f) for f in obj._fieldnames}
        if isinstance(obj, np.ndarray) and obj.dtype.names:  # structured array = struct array
            obj = [{f: r[f] for f in obj.dtype.names} for r in obj.ravel()]
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 2 and obj.shape[1] > 1:
            for row in obj:  # cell array: keep (name, items) rows together
                self.walk(row.tolist(), img, cls_hint)
            return
        if isinstance(obj, dict):
            lower = {str(k).lower(): k for k in obj}
            if "br" in lower:
                cls = None
                for f in ("class", "classname", "label", "name", "food"):
                    if f in lower and isinstance(obj[lower[f]], str):
                        cls = obj[lower[f]]
                        break
                self.found.append((img, cls or cls_hint, obj[lower["br"]]))
                return
            for k, v in obj.items():
                if str(k).startswith("__"):
                    continue
                m = self.match_image(str(k))
                if m:
                    self.walk(v, m, None)
                else:
                    self.walk(v, img, str(k))
            return
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.ravel().tolist()
        if isinstance(obj, (list, tuple)):
            # A string sibling naming an image sets the context. With one such
            # string it applies to all siblings; with several (flattened
            # name, items, name, items, ... lists) each applies to what follows.
            hits = [self.match_image(c) for c in obj]
            n_hits = sum(h is not None for h in hits)
            if n_hits == 1:
                img = next(h for h in hits if h)
            children = []
            for c, h in zip(obj, hits):
                if h is not None:
                    if n_hits > 1:
                        img = h
                    continue
                if not isinstance(c, str):
                    children.append((img, c))
            if children and children[0][0] and all(_is_number_seq(c) for _, c in children):
                for im, c in children:  # class-agnostic polygon list (json re-export)
                    self.found.append((im, None, c))
                return
            for im, c in children:
                self.walk(c, im, cls_hint)


def parse_polygon(br, w: int, h: int) -> np.ndarray | None:
    p = np.asarray(br, dtype=np.float64)
    if p.ndim == 1:
        if p.size % 2:
            return None
        p = p.reshape(-1, 2)            # x1, y1, x2, y2, ...
    elif p.ndim == 2 and p.shape[0] == 2 and p.shape[1] != 2:
        p = p.T                         # 2xN -> Nx2
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return None
    # MATLAB exports are sometimes (row, col) = (y, x); detect obvious transposition
    if (p[:, 0].max() > w * 1.05 and p[:, 0].max() <= h * 1.05 and p[:, 1].max() <= w * 1.05):
        p = p[:, ::-1]
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    if cv2.contourArea(p.astype(np.float32)) < 16:
        return None
    return p


def read_split(path: Path) -> list[str]:
    if path.suffix.lower() == ".txt":
        return [Path(l.strip()).stem for l in path.read_text().splitlines() if l.strip()]
    if path.suffix.lower() == ".json":
        return [Path(str(n)).stem for n in json.loads(path.read_text())]
    from scipy.io import loadmat
    d = loadmat(str(path), simplify_cells=True)
    names: list[str] = []

    def rec(o):
        if isinstance(o, str):
            names.append(Path(o.strip()).stem)
        elif isinstance(o, dict):
            for k, v in o.items():
                if not str(k).startswith("__"):
                    rec(v)
        elif isinstance(o, (list, tuple)) or (isinstance(o, np.ndarray) and o.dtype == object):
            for c in (o.ravel().tolist() if isinstance(o, np.ndarray) else o):
                rec(c)
    rec(d)
    return names


def exif_orientation(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.getexif().get(274)
    except Exception:  # noqa: BLE001 - no PIL or no EXIF
        return None


def image_size(path: Path) -> tuple[int, int]:
    """(h, w) from the file header when PIL is available (fast), else by decoding.
    EXIF orientation is ignored on purpose: OpenCV's imread and the annotations
    both use the raw pixel grid."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            return h, w
    except ImportError:
        return cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).shape[:2]


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.strip().lower())
    return s.strip("_") or "unknown"


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="UNIMIB2016 folder")
    ap.add_argument("--ann", type=Path, help="annotation file (default: ROOT/annotations.mat); "
                    "may be given several times for json train/test re-exports", action="append")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--task", choices=["segment", "detect"], default="segment")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of train used for val")
    ap.add_argument("--test-frac", type=float, default=0.2, help="only used when no official split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1600,
                    help="re-encode images with this longest side (0 = keep size). Images are always "
                         "re-encoded when they carry an EXIF rotation tag, see below")
    ap.add_argument("--copy", action="store_true", help="copy unrotated full-size images instead of symlinking")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--crop-pad", type=float, default=0.08)
    ap.add_argument("--no-mask-bg", action="store_true", help="keep background in crops")
    ap.add_argument("--inspect", action="store_true", help="print annotation structure and exit")
    ap.add_argument("--train-split", type=Path, default=None, help="TrainingSet.mat/.txt (default: auto)")
    ap.add_argument("--test-split", type=Path, default=None, help="TestSet.mat/.txt (default: auto)")
    ap.add_argument("--export-json", type=Path, default=None, metavar="DIR",
                    help="only decode annotations.mat + split .mat files into DIR/annotations.json and "
                         "DIR/{TrainingSet,TestSet}.txt (for machines without scipy), then exit")
    ap.add_argument("--images", type=Path, default=None,
                    help="image folder (default: ROOT/images, else ROOT/original as in the Kaggle copy)")
    ap.add_argument("--class-list", type=Path, default=None,
                    help="restrict to a class list .mat/.txt, e.g. ROOT/final_food_list.mat (65 classes used in "
                         "the paper); instances of other classes are dropped from crops but kept for the detector")
    ap.add_argument("--class-map", type=Path, default=None,
                    help="CSV with columns italian,english[,display_name] (e.g. tools/class_names_en.csv): "
                         "renames classes in crops/, instances.json, classes.txt and the price template; "
                         "raw_class keeps the official name")
    args = ap.parse_args()

    ann_files = args.ann or [args.root / "annotations.mat"]

    if args.export_json:
        args.export_json.mkdir(parents=True, exist_ok=True)

        def plain(o):
            if isinstance(o, dict):
                return {str(k): plain(v) for k, v in o.items() if not str(k).startswith("__")}
            if isinstance(o, np.ndarray):
                return plain(o.tolist()) if o.dtype == object else o.tolist()
            if isinstance(o, (list, tuple)):
                return [plain(c) for c in o]
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            if isinstance(o, bytes):
                return o.decode(errors="replace")
            return o
        merged = {}
        for f in ann_files:
            merged[f.stem] = plain(load_any(f))
        (args.export_json / "annotations.json").write_text(json.dumps(merged))
        for name in ("TrainingSet", "TestSet"):
            for d in (args.root / "split", args.root):
                if (d / f"{name}.mat").exists():
                    (args.export_json / f"{name}.txt").write_text("\n".join(read_split(d / f"{name}.mat")) + "\n")
                    break
        print(f"exported to {args.export_json}")
        return

    if args.inspect:
        for f in ann_files:
            print(f"=== {f}")
            inspect(load_any(f))
        return

    img_dir = args.images or next((args.root / d for d in ("images", "original") if (args.root / d).is_dir()),
                                  args.root / "images")
    # strip download-duplicate suffixes such as "20151130_114525(0).jpg"
    images = {re.sub(r"\(\d+\)$", "", p.stem): p for p in sorted(img_dir.rglob("*"))
              if p.suffix.lower() in IMG_EXTS}
    if not images:
        sys.exit(f"no images found under {img_dir}")
    walker = Walker(set(images))
    for f in ann_files:
        walker.walk(load_any(f))
    if not walker.found:
        sys.exit("no instances with a BR polygon were found; run with --inspect and adapt Walker")

    per_image: dict[str, list] = defaultdict(list)
    dropped = Counter()
    sizes = {}
    for img_name, cls, br in walker.found:
        if img_name is None:
            dropped["no_image_context"] += 1
            continue
        if img_name not in sizes:
            sizes[img_name] = image_size(images[img_name])
        h, w = sizes[img_name]
        poly = parse_polygon(br, w, h)
        if poly is None:
            dropped["bad_polygon"] += 1
            continue
        per_image[img_name].append((cls.strip() if isinstance(cls, str) else "food", poly))

    # ---- split
    def find_split(name, override):
        if override:
            return override
        for d in (args.root / "split", args.root):
            for ext in (".mat", ".txt"):
                if (d / f"{name}{ext}").exists():
                    return d / f"{name}{ext}"
        return args.root / "split" / f"{name}.mat"
    tr_file, te_file = find_split("TrainingSet", args.train_split), find_split("TestSet", args.test_split)
    names = sorted(per_image)
    rng = random.Random(args.seed)
    if tr_file.exists() and te_file.exists():
        train = [n for n in read_split(tr_file) if n in per_image]
        test = [n for n in read_split(te_file) if n in per_image]
        unused = set(per_image) - set(train) - set(test)
        print(f"official split: {len(train)} train / {len(test)} test"
              + (f" ({len(unused)} annotated images are in neither list and are left out)" if unused else ""))
    else:
        shuffled = names[:]
        rng.shuffle(shuffled)
        k = int(round(len(shuffled) * args.test_frac))
        test, train = shuffled[:k], shuffled[k:]
        print(f"random split: {len(train)} train / {len(test)} test")
    train = sorted(train)
    rng.shuffle(train)
    k = int(round(len(train) * args.val_frac))
    splits = {"val": sorted(train[:k]), "train": sorted(train[k:]), "test": sorted(test)}

    allowed = None
    if args.class_list:
        if args.class_list.suffix.lower() == ".mat":
            from scipy.io import loadmat
            vals = [v for k, v in loadmat(str(args.class_list), simplify_cells=True).items() if not k.startswith("__")]
            allowed = {safe_name(str(c)) for c in np.ravel(vals[0])}
        else:
            allowed = {safe_name(l) for l in args.class_list.read_text().splitlines() if l.strip()}
        present = {safe_name(c) for items in per_image.values() for c, _ in items}
        print(f"class list: {len(allowed)} classes, {len(allowed & present)} present in annotations; "
              f"dropped from gallery crops: {sorted(present - allowed)}")

    cmap, cdisplay = {}, {}
    if args.class_map:
        with open(args.class_map, newline="") as f:
            for r in csv.DictReader(f):
                cmap[safe_name(r["italian"])] = safe_name(r["english"])
                cdisplay[safe_name(r["english"])] = (r.get("display_name") or "").strip()
        if len(set(cmap.values())) != len(cmap):
            sys.exit(f"{args.class_map}: two classes map to the same name")
        present = {safe_name(c) for items in per_image.values() for c, _ in items}
        if allowed is not None:
            present &= allowed
        unmapped = sorted(present - set(cmap))
        print(f"class map: {len(cmap)} entries; unmapped classes keep their name: {unmapped}")

    def cname(c):
        return cmap.get(safe_name(c), safe_name(c))

    # ---- write
    out = args.out.resolve()
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    instances = []
    class_counts = Counter()
    for split, names_s in splits.items():
        for name in names_s:
            src = images[name]
            dst = out / "images" / split / f"{name}.jpg"
            label_file = out / "labels" / split / f"{name}.txt"
            rh, rw = sizes[name]
            scale = min(1.0, args.max_side / max(rh, rw)) if args.max_side > 0 else 1.0
            h, w = int(round(rh * scale)), int(round(rw * scale))
            # UNIMIB polygons live in the RAW pixel grid (MATLAB imread ignores EXIF), but
            # most photos carry an EXIF rotation tag that OpenCV / PIL / Ultralytics apply
            # on load. Such images are rewritten unrotated and without EXIF so every tool
            # sees the same pixels the annotations refer to.
            reencode = scale < 1.0 or exif_orientation(src) not in (None, 1)
            img = None  # decoded lazily: re-runs skip images whose label + crops already exist

            def load():
                im = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if scale < 1.0:
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                return im
            if reencode:
                if not (dst.exists() and not dst.is_symlink() and label_file.exists()):
                    img = load()
                    tmp = dst.with_suffix(".tmp.jpg")
                    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    os.replace(tmp, dst)
            else:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)  # relative: survives moving
            lines = []
            for k, (cls, poly) in enumerate(per_image[name]):
                poly = poly * scale
                x1, y1 = poly.min(0)
                x2, y2 = poly.max(0)
                if args.task == "segment":
                    approx = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), 1.0, True).reshape(-1, 2)
                    if len(approx) < 3:
                        approx = poly
                    coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in approx)
                    lines.append(f"0 {coords}")
                else:
                    lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                 f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                crop_rel = None
                in_list = allowed is None or safe_name(cls) in allowed
                if not args.no_crops and in_list:
                    cdir = out / "crops" / split / cname(cls)
                    crop_path = cdir / f"{name}_{k}.jpg"
                    if not (crop_path.exists() and label_file.exists()):
                        if img is None:
                            img = load()
                        cdir.mkdir(parents=True, exist_ok=True)
                        crop = crop_instance(img, (x1, y1, x2, y2), poly, pad=args.crop_pad,
                                             mask_bg=not args.no_mask_bg)
                        tmp = crop_path.with_suffix(".tmp.jpg")
                        cv2.imwrite(str(tmp), crop)
                        os.replace(tmp, crop_path)  # atomic: an interrupted run never leaves half a file
                    crop_rel = str(crop_path.relative_to(out))
                instances.append({
                    "image": name, "file": str(dst.relative_to(out)), "scale": scale,
                    "split": split, "class": cname(cls), "raw_class": cls,
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "polygon": np.round(poly, 1).tolist(), "crop": crop_rel, "in_class_list": in_list,
                })
                class_counts[(split, cname(cls))] += 1
            label_file.write_text("\n".join(lines) + "\n")  # written last = marks the image as done

    classes = sorted({i["class"] for i in instances})
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    (out / "instances.json").write_text(json.dumps(instances))
    (out / "data.yaml").write_text(
        f"# class-agnostic food detector generated by convert_unimib.py\n"
        f"# no 'path:' key -> ultralytics resolves the folders relative to this file\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: food\n")
    with open(out / "prices_template.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "price", "weight_g", "display_name"])
        for c in classes:
            raw = next(i["raw_class"] for i in instances if i["class"] == c)
            wr.writerow([c, "", "", cdisplay.get(c) or raw])

    print(f"images: " + ", ".join(f"{s}={len(v)}" for s, v in splits.items()))
    print(f"instances: {len(instances)}  classes: {len(classes)}  dropped: {dict(dropped)}")
    missing = [c for c in classes if class_counts[("train", c)] == 0]
    if missing:
        print(f"WARNING: {len(missing)} classes have no train crops (gallery cannot recognise them): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
