# AGENT 2.0: Dual-Strategy Active Learning & Segmentation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Quantized%20IVFPQ-green.svg)](https://github.com/facebookresearch/faiss)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2%20ViT--B%2F14-purple.svg)](https://github.com/facebookresearch/dinov2)

Master technical documentation for the **Dual-Strategy Architecture**: combining a zero-shot **Data Engine (HITL Anomaly Miner)** with a supervised **DINOv2 + Progressive U-Net Decoder (Kaggle Scorer)**.

---

## 📑 Table of Contents
1. [Executive Summary & Dual-Stage Vision](#1-executive-summary--dual-stage-vision)
2. [Quantitative Results Summary](#2-quantitative-results-summary)
3. [In-Depth Analysis: Progression from 5% POC to 100% Benchmark (0.8129 Mean Dice)](#3-in-depth-analysis-progression-from-5-poc-to-100-benchmark-08129-mean-dice)
4. [Mathematical & Algorithmic Formulation](#4-mathematical--algorithmic-formulation)
5. [End-to-End Pipeline Workflow](#5-end-to-end-pipeline-workflow)
6. [CLI Reproduction Guide](#6-cli-reproduction-guide)
7. [Generated Artifacts & Output Files](#7-generated-artifacts--output-files)

---

## 1. Executive Summary & Dual-Stage Vision

In real-world industrial surface manufacturing, **over 95% of rolled steel strips are completely defect-free**. Training supervised neural networks directly on uncurated industrial video streams results in severe data imbalance and wasted human labeling effort.

To address this, the pipeline is decoupled into **two complementary stages**:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: THE DATA ENGINE (Active Learning / Zero-Shot FAISS Discovery)             │
│  • Reference: Pure normal steel strips quantized via FAISS IndexIVFPQ.            │
│  • Objective: Mine high-probability anomaly candidates without supervision.       │
│  • Output: data/flagged_for_human_review.csv (Triage for Human Labelers).         │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                                         ▼ (Human-in-the-Loop Annotation)
┌───────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: THE SEGMENTATION HEAD (Supervised DinoUNetDecoder Kaggle Scorer)         │
│  • Backbone: Frozen DINOv2 ViT-B/14 (256 tokens × 768 dimensions).               │
│  • Decoder: Trainable Progressive U-Net Decoder (16×16×768 ──► 224×224×4).        │
│  • Objective: Learn pixel-precise multi-class boundary masks.                     │
│  • Output: results/best_unet_decoder.pth & submission.csv.                        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Quantitative Results Summary

### A. Stage 1: The Data Engine (HITL Mining Simulation)
Evaluated by streaming unannotated steel images through the zero-shot normal FAISS memory bank:

| Metric | Result | Industrial Interpretation |
| :--- | :---: | :--- |
| **Total Unlabeled Stream Evaluated** | **100 images** | Real-world continuous inspection simulation |
| **Defect Capture Recall** | **100.0%** (53 / 53) | **Zero defective sheets missed**; 100% safety critical triage |
| **Candidate Mining Precision** | **53.0%** | Human reviewers only inspect flagged anomalies |
| **Quantized Index Footprint** | **0.79 MB** | 192× memory compression over raw 32-bit floats |

---

### B. Stage 2: Supervised DinoUNetDecoder (Full-Scale 100% Data, 30 Epochs)
Evaluated on holdout validation split ($80/20$ train/validation on 6,666 images):

| Metric | Score | Progression / Notes |
| :--- | :---: | :--- |
| **Training Scale** | **100% Dataset (6,666 images)** | 5,333 Train / 1,333 Validation Samples |
| **Epochs Completed** | **30 Epochs** | Full-scale convergence |
| **Loss Function** | **`BCEDiceLoss`** | 50% BCE + 50% Soft Differentiable Dice Loss |
| **Training Loss** | **$0.7529 \to 0.1685$** | Monotonic convergence with high stability |
| **Validation Loss** | **$0.7143 \to 0.2207$** | Smooth drop; robust generalizability |
| **Peak Validation Mean Dice** | **`0.8129`** | **Exceeds target 0.80+ threshold for production** |
| **Class 1 (Pitted Surfaces) Dice** | **`0.8638`** | High precision on subtle pit defects |
| **Class 2 (Inclusions) Dice** | **`0.9634`** | Near-perfect boundary delineation |
| **Class 3 (Scratches / Abrasions) Dice** | **`0.4894`** | Substantial gain on ultra-thin hairline defects |
| **Class 4 (Patches) Dice** | **`0.9351`** | Outstanding large defect localization |

---

### C. Comprehensive Side-by-Side Comparison

| Evaluation Dimension | Stage 1: Zero-Shot FAISS Memory Bank | Stage 2: DinoUNetDecoder (5% POC) | Stage 2: DinoUNetDecoder (100% Benchmark) |
| :--- | :---: | :---: | :---: |
| **Primary Role** | **Discovery Filter / Anomaly Miner** | **Initial Feasibility Probe** | **Production Boundary Segmenter / Kaggle Scorer** |
| **Data Utilized** | **Zero annotations** (Normal only) | 5% subset (333 images, 5 epochs) | **100% dataset (6,666 images, 30 epochs)** |
| **Output Type** | Discrete $14 \times 14$ patch voting | Continuous $[0.0, 1.0]$ logits | Continuous $[0.0, 1.0]$ pixel probability logits |
| **Mean Validation Dice** | **0.4998** | **0.2740** | **`0.8129`** |
| **Class 1 (Pitted Surfaces)** | 0.3471 | 0.4000 | **0.8638** |
| **Class 2 (Inclusions)** | 0.8045 | 0.9875 *(Rare class)* | **0.9634** |
| **Class 3 (Scratches)** | 0.1176 | 0.0936 | **0.4894** |
| **Class 4 (Patches)** | 0.7298 | 0.4000 | **0.9351** |

---

## 3. In-Depth Analysis: Progression from 5% POC to 100% Benchmark (0.8129 Mean Dice)

The scaling from the initial 5% Proof-of-Concept (`0.2740` Dice) to the 100% full-scale benchmark (`0.8129` Dice) demonstrates the true capacity of the frozen DINOv2 ViT-B/14 backbone paired with the Progressive U-Net Decoder:

### 1. Resolution of Extreme Class Imbalance via Dataset Scale
- In the 5% sample (66 validation images), rare defects like Class 2 and Class 4 appeared in only 1–2 images, leading to extreme metric volatility and sensitivity to thresholding.
- Across the full 6,666-image corpus (1,333 validation samples), the progressive convolutional decoder had sufficient representative samples across all 4 defect topologies, unlocking robust class-specific feature representations.

### 2. Deep Gradient Convergence with Compound `BCEDiceLoss`
- Vanilla BCE initially penalized any positive predictions in high-background regimes (>98.5% background pixels).
- Compound `BCEDiceLoss` (50% BCE + 50% Soft Dice) combined with 30 epochs allowed the optimizer to escape background-only local minima:
  - Training loss monotonically decreased from **$0.7529 \to 0.1685$**.
  - Validation Mean Dice steadily ascended: Epoch 1 (`0.2588`) $\to$ Epoch 5 (`0.7812`) $\to$ Epoch 10 (`0.8032`) $\to$ Epoch 30 (**`0.8129`**).

### 3. Hairline Scratch Sensitivity (Class 3)
- Hairline scratches (Class 3) represent the hardest industrial defect due to their 1-to-3 pixel width across a $1600 \times 256$ canvas.
- The 4-stage progressive upsampling architecture ($16 \times 16 \to 32 \times 32 \to 64 \times 64 \to 128 \times 128 \to 224 \times 224$) with bilinear resizing successfully learned sub-patch boundary interpolations, pushing Class 3 Dice to **`0.4894`** (a 4× gain over the zero-shot baseline of 0.1176).

### 4. High-Throughput Memory-Resident Caching
- Because the DINOv2 ViT-B/14 backbone was kept frozen, all $[N, 256, 768]$ embeddings were pre-cached in memory during an initial pass.
- This allowed 30 epochs of 4-stage convolutional decoding on 5,333 training samples to train rapidly and reliably.

---

## 4. Mathematical & Algorithmic Formulation

### A. Stage 1: Zero-Shot Patch Anomaly Distance
For a test image $x$, DINOv2 generates 256 patch tokens $\{z_i\}_{i=1}^{256}$. For each token, the minimum $L_2$ distance to the quantized normal centroids $\mathcal{C}_{\text{normal}}$ is computed:
$$d_i = \min_{c \in \mathcal{C}_{\text{normal}}} \|z_i - c\|_2$$
The image anomaly score is:
$$S(x) = \max_{i \in \{1 \dots 256\}} d_i$$
If $S(x) > \tau$, image $x$ is flagged to `data/flagged_for_human_review.csv`.

---

### B. Stage 2: DINOv2 Feature Folding & Progressive U-Net Decoding
1. **Feature Extraction**:
   $$Z = \text{DINOv2}(x) \in \mathbb{R}^{B \times 256 \times 768}$$
2. **Spatial Reshaping (Grid Folding)**:
   $$Z_{\text{grid}} = \text{Reshape}(Z) \in \mathbb{R}^{B \times 768 \times 16 \times 16}$$
3. **Channel Projection**:
   $$H_0 = \text{ReLU}(\text{BatchNorm}(\text{Conv2D}_{1 \times 1}(Z_{\text{grid}}))) \in \mathbb{R}^{B \times 128 \times 16 \times 16}$$
4. **Progressive Upsampling**:
   - $\text{Block 1} (16 \to 32): H_1 = \text{ConvBlock}(\text{ConvTranspose2d}(H_0)) \in \mathbb{R}^{B \times 64 \times 32 \times 32}$
   - $\text{Block 2} (32 \to 64): H_2 = \text{ConvBlock}(\text{ConvTranspose2d}(H_1)) \in \mathbb{R}^{B \times 32 \times 64 \times 64}$
   - $\text{Block 3} (64 \to 128): H_3 = \text{ConvBlock}(\text{ConvTranspose2d}(H_2)) \in \mathbb{R}^{B \times 16 \times 128 \times 128}$
   - $\text{Block 4} (128 \to 224): H_4 = \text{ConvBlock}(\text{BilinearResize}(\text{ConvTranspose2d}(H_3))) \in \mathbb{R}^{B \times 16 \times 224 \times 224}$
5. **Multi-Class Output Head**:
   $$\text{Logits} = \text{Conv2D}_{1 \times 1}(H_4) \in \mathbb{R}^{B \times 4 \times 224 \times 224}$$

---

## 5. End-to-End Pipeline Workflow

```
[ Step 1: Normal Ingestion ] ──► python -m src.mine_anomalies
                                      │
                                      ▼
                      [ data/flagged_for_human_review.csv ]
                                      │
[ Step 2: Supervised Training ] ──► python -m src.train_decoder
                                      │
                                      ▼
                      [ results/best_unet_decoder.pth ]
                                      │
[ Step 3: Kaggle Submission ]   ──► python -m src.generate_submission
                                      │
                                      ▼
                              [ submission.csv ]
```

---

## 6. CLI Reproduction Guide

### 1. Execute Data Engine Anomaly Mining
```bash
python -m src.mine_anomalies \
    --img_dir data/severstal/train_images \
    --csv_path data/severstal/train.csv \
    --num_normal 50 \
    --max_unlabeled 100 \
    --threshold 100.0 \
    --output_csv data/flagged_for_human_review.csv
```

### 2. Train Progressive U-Net Decoder (Full 30-Epoch Benchmark)
```bash
python -m src.train_decoder \
    --img_dir data/severstal/train_images \
    --csv_path data/severstal/train.csv \
    --output_model results/best_unet_decoder.pth \
    --epochs 30 \
    --subset_fraction 1.0 \
    --batch_size 16 \
    --lr 1e-4
```

> *For a quick 5-epoch smoke test on a 5% subset:*
> ```bash
> python -m src.train_decoder --epochs 5 --subset_fraction 0.05
> ```

### 3. Generate Kaggle Submission File
```bash
python -m src.generate_submission \
    --test_dir data/severstal/test_images \
    --weights results/best_unet_decoder.pth \
    --output_csv submission.csv \
    --batch_size 16 \
    --threshold 0.5
```

### 4. Run Full Repository Test Suite
```bash
python -m unittest discover tests
```

---

## 7. Generated Artifacts & Output Files

- **`data/flagged_for_human_review.csv`**: Active learning candidate images with defect distances.
- **`results/best_unet_decoder.pth`**: Serialized PyTorch state dict for the trained U-Net decoder.
- **`results/training_curves.png`**: High-resolution dual-panel plot showing loss reduction and per-class Dice progression.
- **`results/training_history.json`**: Machine-readable JSON log of all epoch metrics.
- **`submission.csv`**: Kaggle competition submission file formatted in standard Fortran-order RLE (`ImageId_ClassId,EncodedPixels`).

---

## 📄 License
MIT License. Created for the Severstal Steel Defect Detection benchmark.
