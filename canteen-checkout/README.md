# Canteen vision checkout: YOLO detection + vector-retrieval recognition

A two-stage canteen tray checkout prototype built on UNIMIB2016:

```
tray photo ──► YOLOv8/YOLO11-seg (a single class: food) ──► box + outline per dish
               │
               ├─► crop along the outline, mask the background ──► DINOv2 features (optional ArcFace fine-tune) ──► L2-normalised vector
               │                                                      │
               │                         gallery.npz (FAISS / numpy cosine kNN) ◄─┘
               │                                                      │
               └─► decision: accept / uncertain (ask the customer) / unknown ──► price list ──► weight cross-check ──► receipt
```

**Why split it this way**: the detector only answers "how many dishes, and where"; it knows nothing about the menu. "Which dish is this" is left to vector retrieval. So **adding a dish only takes a few photos added to the gallery, with no model retraining** (`build_gallery.py --append`).

### What is borrowed from lannguyen0910/food-recognition
That repository uses YOLOv5 for detection, EfficientNet to re-classify the detections, and TTA + Weighted Boxes Fusion to merge results. This project keeps the "detect first, then recognise" structure and **flip TTA + WBF** (`checkout/detector.py`, `checkout/boxes.py`), with three changes:
1. The detector is Ultralytics YOLOv8/YOLO11, trained as a **class-agnostic** single-class detector (`single_cls=True`).
2. The fixed-class re-classifier is replaced by **vector retrieval** with open-set rejection (unknown).
3. Checkout-specific outputs are added: confirmation of uncertain items, a price list, a weight check, and a "silent error rate" evaluation.

---

## Layout

| File | Purpose |
|---|---|
| `tools/convert_unimib.py` | UNIMIB2016 → YOLO-seg/detect labels, `data.yaml`, per-class crops, `instances.json`, price template |
| `tools/class_names_en.csv` | Italian → English class names (used with `--class-map`) |
| `tools/relabel_gallery.py` | Rename the classes of an existing `gallery.npz` without re-embedding |
| `train_detector.py` | Train the class-agnostic detector (YOLO11 / YOLOv8, seg or detect) |
| `finetune_embedder.py` | Optional: ArcFace fine-tune of the retrieval backbone; the validation metric is retrieval top-1 |
| `build_gallery.py` | Build the gallery from crops; `--append` adds dishes |
| `evaluate.py` | Three-level evaluation (detection / recognition / end to end); `--calibrate` picks thresholds |
| `demo.py` + `configs/checkout.yaml` | Receipt for one image or a whole folder |
| `colab_checkout.ipynb` | The full workflow on Google Colab (GPU) |
| `checkout/` | Library code: cropping, detector wrapper, embedder, vector index, pipeline |
| `tests/smoke_test.py` | Smoke test without torch (synthetic data) |
| `tools/make_synthetic.py` | Synthetic data with the same structure as UNIMIB2016, for tests |

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # ultralytics installs torch; with a GPU, install the matching CUDA build from pytorch.org first
python tests/smoke_test.py            # ~10 s; checks conversion, index, pipeline and evaluation end to end
```

## 1. Prepare UNIMIB2016

Request/download UNIMIB2016 from the IVL lab at the University of Milano-Bicocca (research use only; follow its licence). After extracting you should have:

```
UNIMIB2016/
  images/*.jpg
  annotations.mat
  split/TrainingSet.mat  split/TestSet.mat
```

```bash
# look at the annotation structure first (the nesting of annotations.mat differs slightly between releases)
python tools/convert_unimib.py --root /path/UNIMIB2016 --out data/unimib_yolo --inspect

# convert: instance-segmentation labels (polygons) by default, 10% of train held out as val, official test split
python tools/convert_unimib.py --root /path/UNIMIB2016 --out data/unimib_yolo --class-map tools/class_names_en.csv
```

**Kaggle copy (`archive/original/` + the `.mat` files at the top level)**: the official `annotations.mat` stores a MATLAB `containers.Map` object, which the converter decodes directly (needs scipy). If the machine running the conversion has no scipy, export JSON on a machine that has it, then convert from the JSON:

```bash
python tools/convert_unimib.py --root archive --export-json archive/annotations_json   # needs scipy
python tools/convert_unimib.py --root archive --images archive/original \
    --ann archive/annotations_json/annotations.json \
    --train-split archive/annotations_json/TrainingSet.txt --test-split archive/annotations_json/TestSet.txt \
    --class-map tools/class_names_en.csv --out archive/unimib_yolo
```

Actual result: the official split has 650 train / 360 test images, and 17 annotated images are in neither list and are excluded. That leaves 1,010 images, 3,561 instances and **65 classes** (matching the paper; the other 8 classes appear only in those 17 images). 6 classes have fewer than 5 training samples.

**Class names**: UNIMIB class names are Italian (`banane`, `pane`, ...). With `--class-map tools/class_names_en.csv` the converter writes English names to `crops/`, `instances.json`, `classes.txt` and `prices_template.csv` (whose `display_name` column gets a readable English name); `raw_class` in `instances.json` keeps the official Italian name. Without `--class-map` the Italian names are kept. A gallery built with the Italian names can be switched without re-embedding: `python tools/relabel_gallery.py --gallery gallery.npz --out gallery_en.npz`.

**EXIF orientation (important)**: 925 of the 1,027 photos carry an EXIF rotation tag, but the UNIMIB polygons are annotated in the **raw pixel grid** (MATLAB's `imread` ignores EXIF). OpenCV, PIL and Ultralytics all apply EXIF rotation by default, which would misalign the labels. The converter re-encodes these images in the raw pixel orientation without EXIF, and by default also resizes them to a 1600 px long side (`--max-side`). Download-duplicate suffixes such as `(0)` in file names are stripped automatically.

**Some items are not annotated**: UNIMIB only annotates some of the food; bread, ham, packaged food and the like on a tray often have no outline. The detector learns them as background, so item counts will miss them. Before real deployment, add the missing annotations or fine-tune on your own data.

The converter does not hard-code the `.mat` hierarchy. It walks the whole structure and collects every node with a `BR` (boundary polygon) field, infers the image name from the nearest key or sibling string, and the class from a `class` field or the enclosing key. It also accepts the `annotations/{train,test}.json` export from gist-ailab/Food-Instance-Segmentation (`--ann train.json --ann test.json`; those have no class names, so only the detector can be trained).

**Check the printed statistics**: UNIMIB2016 is published as 1,027 trays, 73 classes and about 3,616 instances. If the numbers are clearly off, inspect the structure with `--inspect` and adjust `Walker`. The end of the output warns about classes with no train samples (the gallery cannot recognise them).

## 2. Train the detector

```bash
python train_detector.py --data data/unimib_yolo/data.yaml --model yolo11s-seg.pt --epochs 100 --batch 16 --device 0
# Apple silicon Mac: --device mps --batch 8 --workers 2 (without a GPU start with yolo11n-seg.pt and --imgsz 640)
# or YOLOv8: --model yolov8s-seg.pt; boxes only: convert with --task detect and use yolo11s.pt
```

Augmentation is tuned for top-down tray photos: rotation and vertical flips are allowed, hue jitter is almost off (colour is a key cue for telling dishes apart), and mixup is not used. At the end of training the model is evaluated on val/test and the result is written to `eval_report.json`.

Seg models are recommended: the outline masks out the plate and neighbouring dishes when cropping, which helps recognition noticeably; in UNIMIB, dishes often share a plate.

## 3. Build the gallery (recognition)

```bash
# pretrained DINOv2 features as-is
python build_gallery.py --crops data/unimib_yolo/crops/train --out gallery.npz

# optional: ArcFace fine-tune (UNIMIB classes are long-tailed, --balanced recommended)
python finetune_embedder.py --train data/unimib_yolo/crops/train --val data/unimib_yolo/crops/val \
    --out runs/embedder --epochs 20 --balanced
python build_gallery.py --crops data/unimib_yolo/crops/train --checkpoint runs/embedder/best.pt --out gallery_ft.npz
```

Adding dishes (no training):

```bash
# new_dishes/<dish_name>/*.jpg, can be whole tray photos: --detector crops them with the detector first, so they match runtime crops
python build_gallery.py --append gallery.npz --crops new_dishes/ --detector runs/detector/unimib/weights/best.pt --out gallery.npz
```

The folder name is the class name: an existing class name (e.g. `banana`) adds reference photos to that class, a new name creates a class.

The gallery records the embedder configuration it was built with; `--append` refuses to run if the embedder differs (mixing vectors from different models makes retrieval meaningless).

## 4. Threshold calibration and evaluation

```bash
# calibrate on val: the thresholds with the highest accept rate subject to "precision of accepted items >= 98%"
python evaluate.py --dataset data/unimib_yolo --gallery gallery.npz --detector oracle --split val --calibrate 0.98

# evaluate end to end on test with the calibrated thresholds
python evaluate.py --dataset data/unimib_yolo --gallery gallery.npz \
    --detector runs/detector/unimib/weights/best.pt --split test \
    --accept-sim 0.xx --margin 0.xx --out report_test.json --save-vis vis_test/
```

The report has three levels:

- **detection**: precision and recall at IoU >= 0.5, and the share of trays with exactly the right item count.
- **recognition_on_gt_crops**: recognition alone, on ground-truth crops (no detection error): top-1/top-3, accept rate, precision of accepted items, most common confusions, worst classes.
- **end_to_end**: per tray. `auto_correct_rate` is the share of fully correct receipts with no human involved; **`silent_error_rate` is the share of wrong receipts that the system did not flag for review**, split into overcharge and undercharge.

For a checkout system `silent_error_rate` is the key metric: an error flagged for review is only a usability issue, an unnoticed error is a real loss.

## 5. Run

```bash
cp data/unimib_yolo/prices_template.csv data/prices.csv    # fill in prices; weight_g (standard portion weight) is optional
python demo.py --config configs/checkout.yaml --image tray.jpg --weight 540 --save out.jpg
```

Each item on the receipt has one of three states: `accept` (priced automatically), `uncertain` (the top candidates are shown for the customer to pick), `unknown` (not on the menu, not food, or a false detection). `needs_review` is set if any of these holds: an uncertain or unrecognised item, no dish detected, a recognised dish with no price, or a measured weight that deviates from the sum of standard weights by more than `weight_tolerance`.

---

## For red-team testing

The system is designed to serve as a local target, with every component under your control:

- **Attack surfaces**: the detector (make a dish "disappear", i.e. a missed detection → undercharge), embedding and retrieval (make a dish recognised as a different, cheaper one → wrong charge), thresholds and review logic (make a wrong result pass with high confidence → silent error).
- **How to measure**: apply the perturbation or physical condition you want to test to the test-split images, run `evaluate.py`, and compare against the report on clean images. Look at whether `silent_error_rate` and `precision_of_accepted` go up, not just whether top-1 goes down.
- **Built-in defence layers, each can be switched off for ablation**: open-set rejection (`unknown_sim`), the margin constraint (`margin`), the weight cross-check (`--weight`), outline-masked cropping (`mask_bg`), detector TTA.
- Defences worth adding: adversarial training, patch-detection preprocessing such as local gradient smoothing, multi-camera consistency checks.

Only test systems you built yourself or are authorised to test.

## Known limitations

- UNIMIB2016 has only about 1k images and 73 long-tailed classes; some classes have single-digit sample counts. Top-1 depends on the split, so report the mean over several random splits.
- Each dish's gallery samples come from the same canteen and the same batch. A different canteen, tableware or lighting requires new reference photos and recalibrated thresholds.
- The dataset contains Western (Italian) dishes; other cuisines need their own data, and the converter supports the same polygon annotation format.
