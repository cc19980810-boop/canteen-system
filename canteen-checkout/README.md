# 食堂视觉结算：YOLO 检测 + 向量检索识别

在 UNIMIB2016 上实现的两阶段食堂托盘结算原型：

```
托盘照片 ──► YOLOv8/YOLO11-seg（只有一个类别：food）──► 每道菜的框 + 轮廓
             │
             ├─► 按轮廓裁剪并遮住背景 ──► DINOv2 特征（可用 ArcFace 微调）──► L2 归一化向量
             │                                                      │
             │                         图库 gallery.npz（FAISS / numpy 余弦 kNN）◄─┘
             │                                                      │
             └─► 判定：accept / uncertain（请顾客确认）/ unknown ──► 价格表 ──► 称重交叉校验 ──► 小票
```

**为什么这样拆分**：检测器只回答“有几道菜、在哪里”，与菜单无关；“这是哪道菜”交给向量检索。所以**新增一道菜只需要拍几张照片加入图库，不需要重新训练任何模型**（`build_gallery.py --append`）。

### 从 lannguyen0910/food-recognition 借鉴的部分
该仓库使用 YOLOv5 做检测、EfficientNet 对检测结果二次分类，并用 TTA + Weighted Boxes Fusion 融合结果。本项目沿用了“先检测、后二次识别”的结构和 **翻转 TTA + WBF**（`checkout/detector.py`、`checkout/boxes.py`），但做了三处改动：
1. 检测器换成 Ultralytics YOLOv8/YOLO11，并改为**类别无关**的单类检测（`single_cls=True`）；
2. 二次识别由固定类别的分类器换成**向量检索**，并加入开放集拒识（unknown）；
3. 增加针对结算场景的输出：不确定项确认、价格表、称重校验，以及“静默错误率”评估。

---

## 目录

| 文件 | 作用 |
|---|---|
| `tools/convert_unimib.py` | UNIMIB2016 → YOLO-seg/detect 标签、`data.yaml`、每类裁剪图、`instances.json`、价格表模板 |
| `train_detector.py` | 训练类别无关检测器（YOLO11 / YOLOv8，seg 或 detect） |
| `finetune_embedder.py` | 可选：ArcFace 微调检索主干，验证指标就是检索 top-1 |
| `build_gallery.py` | 用裁剪图建图库；`--append` 新增菜品 |
| `evaluate.py` | 检测 / 识别 / 端到端三级评估，`--calibrate` 自动选阈值 |
| `demo.py` + `configs/checkout.yaml` | 单张或整个文件夹出小票 |
| `checkout/` | 库代码：裁剪、检测封装、嵌入、向量索引、流水线 |
| `tests/smoke_test.py` | 不依赖 torch 的冒烟测试（合成数据） |
| `tools/make_synthetic.py` | 生成与 UNIMIB2016 同结构的合成数据，用于测试 |

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # ultralytics 会安装 torch；有 GPU 请先按 pytorch.org 装对应 CUDA 版
python tests/smoke_test.py            # 约 10 秒，验证数据转换、索引、流水线、评估都能跑通
```

## 1. 准备 UNIMIB2016

从米兰比可卡大学 IVL 实验室官网申请/下载 UNIMIB2016（仅限研究用途，请遵守其许可）。解压后应有：

```
UNIMIB2016/
  images/*.jpg
  annotations.mat
  split/TrainingSet.mat  split/TestSet.mat
```

```bash
# 先看一下标注结构（annotations.mat 的嵌套方式在不同发布版本中略有差异）
python tools/convert_unimib.py --root /path/UNIMIB2016 --out data/unimib_yolo --inspect

# 转换：默认生成实例分割标签（多边形），train 中划 10% 作 val，test 用官方划分
python tools/convert_unimib.py --root /path/UNIMIB2016 --out data/unimib_yolo
```

**Kaggle 版（`archive/original/` + 根目录下的 `.mat`）**：官方 `annotations.mat` 存的是 MATLAB `containers.Map` 对象，转换器已能直接解码（需要 scipy）。如果运行转换的机器没有 scipy，可以先在有 scipy 的机器上导出 JSON，再用 JSON 转换：

```bash
python tools/convert_unimib.py --root archive --export-json archive/annotations_json   # 需要 scipy
python tools/convert_unimib.py --root archive --images archive/original \
    --ann archive/annotations_json/annotations.json \
    --train-split archive/annotations_json/TrainingSet.txt --test-split archive/annotations_json/TestSet.txt \
    --out archive/unimib_yolo
```

实际结果：官方划分 650 train / 360 test，另有 17 张不在任何列表中而被排除；共 1,010 张、3,561 个实例、**65 类**（与论文一致，其余 8 类只出现在那 17 张里）。其中 6 个类别的训练样本少于 5 个。

**EXIF 方向（重要）**：1,027 张照片中有 925 张带 EXIF 旋转标记，但 UNIMIB 的多边形是按**原始像素**标注的（MATLAB 的 `imread` 忽略 EXIF）。OpenCV、PIL 和 Ultralytics 默认都会按 EXIF 旋转图片，这样标签就会错位。转换器会把这些图片按原始像素方向重新编码并去掉 EXIF，默认同时缩放到最长边 1600 像素（`--max-side`）。文件名中的 `(0)` 这类下载重复后缀也会自动去掉。

**部分物品未标注**：UNIMIB 只标注了部分食物，托盘上的面包、火腿、包装食品等常常没有轮廓。检测器会把它们当作背景来学习，所以用它来“数件数”时会漏掉这类物品。实际部署前需要补标，或者用自己采集的数据微调。

转换器不写死 `.mat` 的层级，而是遍历整个结构、收集所有带 `BR`（边界多边形）字段的节点，从最近的键名或同级字符串推断图像名，从 `class` 字段或上层键名推断类别。它也接受 gist-ailab/Food-Instance-Segmentation 导出的 `annotations/{train,test}.json`（`--ann train.json --ann test.json`，此时没有类别名，只能训练检测器）。

**请核对输出的统计**：UNIMIB2016 公布的是 1,027 张托盘、73 类、约 3,616 个实例。如果数量明显不对，用 `--inspect` 查看结构并调整 `Walker`。输出末尾会警告哪些类别在 train 中没有样本（这些类图库识别不了）。

## 2. 训练检测器

```bash
python train_detector.py --data data/unimib_yolo/data.yaml --model yolo11s-seg.pt --epochs 100 --batch 16 --device 0
# Apple 芯片 Mac：--device mps --batch 8 --workers 2（没有 GPU 时用 yolo11n-seg.pt、--imgsz 640 起步）
# 或 YOLOv8：--model yolov8s-seg.pt；只要框：转换时加 --task detect，模型用 yolo11s.pt
```

增强参数针对俯拍托盘做了调整：允许旋转和上下翻转，但色调抖动几乎关闭（颜色是区分菜品的关键线索），不使用 mixup。训练结束会在 val/test 上评估并写入 `eval_report.json`。

建议使用 seg 模型：轮廓能在裁剪时遮住餐盘和相邻菜品，对识别准确率帮助明显；UNIMIB 中菜品经常挨在同一个盘子里。

## 3. 建图库（识别模块）

```bash
# 直接使用预训练 DINOv2 特征
python build_gallery.py --crops data/unimib_yolo/crops/train --out gallery.npz

# 可选：ArcFace 微调（UNIMIB 类别长尾，建议 --balanced）
python finetune_embedder.py --train data/unimib_yolo/crops/train --val data/unimib_yolo/crops/val \
    --out runs/embedder --epochs 20 --balanced
python build_gallery.py --crops data/unimib_yolo/crops/train --checkpoint runs/embedder/best.pt --out gallery_ft.npz
```

新增菜品（无需训练）：

```bash
# new_dishes/<菜名>/*.jpg，可以是整张托盘照，--detector 会先用检测器裁出来，保证与运行时裁剪一致
python build_gallery.py --append gallery.npz --crops new_dishes/ --detector runs/detector/unimib/weights/best.pt --out gallery.npz
```

图库记录了建库所用的嵌入模型配置，`--append` 时若嵌入模型不一致会拒绝执行（混用不同模型的向量会导致检索结果毫无意义）。

## 4. 阈值校准与评估

```bash
# 在 val 上校准：在“已接受项准确率 ≥ 98%”的约束下，选出接受率最高的阈值
python evaluate.py --dataset data/unimib_yolo --gallery gallery.npz --detector oracle --split val --calibrate 0.98

# 用校准出的阈值在 test 上做端到端评估
python evaluate.py --dataset data/unimib_yolo --gallery gallery.npz \
    --detector runs/detector/unimib/weights/best.pt --split test \
    --accept-sim 0.xx --margin 0.xx --out report_test.json --save-vis vis_test/
```

评估报告分为三级：

- **detection**：IoU≥0.5 时的精确率、召回率，以及“件数完全正确”的托盘比例；
- **recognition_on_gt_crops**：用真值框裁剪，排除检测误差，单独衡量识别：top-1/top-3、接受率、已接受项准确率、最常见的混淆对、最差类别；
- **end_to_end**：按托盘统计。`auto_correct_rate` 表示无需人工、小票完全正确的比例；**`silent_error_rate` 表示小票有错但系统没有要求复核的比例**，分为多收（overcharge）和少收（undercharge）。

对结算系统来说，`silent_error_rate` 是最关键的指标：被标记复核的错误只是体验问题，没被发现的错误才是真正的损失。

## 5. 运行

```bash
cp data/unimib_yolo/prices_template.csv data/prices.csv    # 填写价格，weight_g（标准份重量）可选
python demo.py --config configs/checkout.yaml --image tray.jpg --weight 540 --save out.jpg
```

小票中每道菜有三种状态：`accept`（自动计价）、`uncertain`（列出前几个候选，请顾客点选）、`unknown`（不在菜单中、不是食物或检测误报）。以下任一情况都会设置 `needs_review`：存在不确定或未识别项、没有检测到菜品、已识别的菜缺少价格，或称重与识别结果的标准重量之和偏差超过 `weight_tolerance`。

---

## 用于红队测试

这套系统设计成可以在本地作为靶机，全部组件都在自己的控制下：

- **攻击面对应**：检测器（让菜品“消失”，即漏检 → 少收）、嵌入与检索（让菜品被识别成另一道更便宜的菜 → 错收）、阈值与复核逻辑（让错误结果以高置信度通过 → 静默错误）。
- **衡量方式**：对 test 集图像施加你要测试的扰动或物理条件变化，再运行 `evaluate.py`，与干净图像的报告对比。重点看 `silent_error_rate` 和 `precision_of_accepted` 是否上升，而不只是 top-1 是否下降。
- **已内置的防御层，可以逐一开关做消融实验**：开放集拒识（`unknown_sim`）、分差约束（`margin`）、称重交叉校验（`--weight`）、轮廓遮罩裁剪（`mask_bg`）、检测 TTA。
- 值得补充的防御方向：对抗训练、局部梯度平滑等补丁检测预处理、多视角相机一致性检查。

请只在自己搭建或获得授权的系统上进行测试。

## 已知局限

- UNIMIB2016 只有约 1 千张图、73 类且长尾明显，部分类别只有个位数样本。top-1 数字会受划分影响，建议报告多次随机划分的均值。
- 图库中每道菜的样本来自同一食堂同一批次，换食堂、换餐具或换光照后需要重新拍摄入库，阈值也需要重新校准。
- 该数据集为西式（意大利）菜品；中式食堂需要自行采集数据，转换器支持同样的多边形标注格式。
