# CLAUDE.md — canteen-checkout

Two-stage canteen tray checkout on UNIMIB2016, built as a local target for red-team
evaluation of vision-based self-checkout:

1. class-agnostic YOLO11/YOLOv8-seg detector (single class `food`) → boxes + outlines
2. crop (outline-masked) → DINOv2 embedding (timm, optional ArcFace fine-tune) → kNN over
   `gallery.npz` → accept / uncertain / unknown → price list → weight cross-check

Read `README.md` for the full workflow. The user writes in Chinese; reply in Chinese.

## Layout on this machine

```
~/Downloads/archive/
  original/              1,027 raw UNIMIB2016 photos (3264x2448) — NEVER modify
  annotations.mat, TrainingSet.mat, TestSet.mat, *_food_list.mat   — official files, NEVER modify
  annotations_json/      same annotations decoded to JSON + split .txt + class lists
  unimib_yolo/           converted dataset (already generated, see below)
  canteen-checkout/      this project
  colab-upload/          zips + notebook for Google Colab (zip unpacks to canteen_checkout/)
```

`unimib_yolo/` = output of `tools/convert_unimib.py` with the official split:
585 train / 65 val / 360 test images, 3,561 instances, 65 classes, images re-encoded at
1600 px long side. `data.yaml` has no `path:` key on purpose (Ultralytics resolves the
folders relative to the yaml). `crops/{train,val,test}/<class>/` hold gallery crops;
`instances.json` holds every GT instance (boxes/polygons in the 1600 px image coordinates).
Class names are English, generated with `--class-map tools/class_names_en.csv`; `raw_class`
keeps the official Italian name. `tools/relabel_gallery.py` converts galleries built with the
Italian names.

## Hard-won facts — do not regress

- **EXIF orientation**: 925/1027 raw photos have an EXIF rotation tag, but the UNIMIB
  polygons are in the RAW pixel grid (MATLAB imread ignores EXIF). OpenCV, PIL and
  Ultralytics all apply EXIF by default. The converter therefore rewrites images unrotated
  and without EXIF. Never train or crop from `original/` directly; if you must read it, use
  `cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION`.
- `annotations.mat` is a MATLAB `containers.Map` (MCOS object); scipy exposes it only as
  `__function_workspace__`. `read_containers_map()` in the converter decodes it. Tuple layout
  per `demo.m`: category, class, item name, boundary type, BR polygon `[x1,y1,...]`, bbox.
- 21 files are named like `20151130_114525(0).jpg`; the converter strips the `(n)` suffix.
- 17 annotated images are in neither official split and are excluded; the 8 classes that
  appear only in them are why 65 (not 73) classes remain — this matches the paper.
- UNIMIB does not annotate every item on a tray (bread rolls, cold cuts, packaged food are
  often unlabelled), so the detector learns them as background. Keep this in mind when
  reading detection recall / count accuracy.
- 6 classes have < 5 training crops; per-class numbers for them are noise.
- Gallery crops and runtime crops must come from the same `checkout.crop.crop_instance`
  settings (pad 0.08, mask_bg True) or retrieval silently degrades.
- `build_gallery.py --append` refuses to mix embedder configs; keep it that way.

## Status

- Verified end to end without torch: converter (on the real data), vector index, decision
  logic, WBF, YoloDetector flip-TTA coordinate mapping (stubbed Ultralytics), pipeline,
  evaluate.py. `python tests/smoke_test.py` must stay green after any change (~10 s; needs scipy).
- NOT yet run anywhere: `train_detector.py`, `TimmEmbedder`, `finetune_embedder.py`
  (written against ultralytics>=8.3, timm>=1.0, torch>=2). Expect small API fixes on first run.

## Environment (Apple silicon Mac)

```bash
cd ~/Downloads/archive/canteen-checkout
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch from PyPI includes MPS support
python tests/smoke_test.py
```

Use `--device mps` for Ultralytics and `device="mps"` for timm; fall back to CPU if an op is
unsupported (`PYTORCH_ENABLE_MPS_FALLBACK=1`). Keep `--workers` ≤ 2 on a MacBook Air, and
prefer `yolo11n-seg.pt` / `--imgsz 640` first to get a quick baseline before larger models.
Long trainings: run them in the background with output to a log file and poll the log.

## Workflow

```bash
# 1. detector
python train_detector.py --data ../unimib_yolo/data.yaml --model yolo11n-seg.pt \
    --device mps --batch 8 --workers 2 --epochs 100
# 2. gallery (pretrained DINOv2 ViT-S/14)
python build_gallery.py --crops ../unimib_yolo/crops/train --out gallery.npz --device mps
# 3. calibrate thresholds on val (GT boxes), then evaluate end to end on test
python evaluate.py --dataset ../unimib_yolo --gallery gallery.npz --detector oracle --split val --calibrate 0.98
python evaluate.py --dataset ../unimib_yolo --gallery gallery.npz \
    --detector runs/detector/unimib/weights/best.pt --split test \
    --accept-sim <from step 3> --margin <from step 3> --out report_test.json --save-vis vis_test/
# 4. optional: ArcFace fine-tune, rebuild gallery with --checkpoint, re-run 3
python finetune_embedder.py --train ../unimib_yolo/crops/train --val ../unimib_yolo/crops/val \
    --out runs/embedder --balanced --device mps
```

Key metric: `end_to_end.silent_error_rate` (wrong receipt not flagged for review). Always
report it next to top-1; calibrate on val, report on test, never tune on test.

## Red-team scope

This system exists to be attacked locally. Robustness experiments go through
`evaluate.py` on transformed copies of the test split, comparing reports against the clean
baseline (silent_error_rate, precision_of_accepted, detection recall). Built-in defences to
ablate: `unknown_sim`, `margin`, weight cross-check, `mask_bg`, detector TTA. Only test
systems the user owns or is authorised to test.

## Conventions

- Library code in `checkout/` stays importable without torch (lazy imports).
- New outputs go under `runs/`, never into `original/` or `unimib_yolo/` (regenerate the
  dataset with the converter instead of editing it by hand).
