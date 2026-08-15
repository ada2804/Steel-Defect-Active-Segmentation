# Severstal Steel Defect Detection (Zero-Shot IVFPQ Pipeline)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Quantized%20IVFPQ-green.svg)](https://github.com/facebookresearch/faiss)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2%20ViT--B%2F14-purple.svg)](https://github.com/facebookresearch/dinov2)

A modular, high-throughput, memory-efficient zero-shot anomaly detection and multi-class segmentation pipeline for industrial steel surface inspection, evaluated on the **Severstal Steel Defect Detection** benchmark.

---

## 📌 Architecture Overview

```
                         [ Input Steel Strip (1600 x 256) ]
                                         │
                                         ▼
                             [ Resize to 224 x 224 ]
                                         │
                                         ▼
                         [ Frozen DINOv2 (ViT-B/14) ]
                                         │
                                         ▼
                     [ 256 Patch Embeddings (768-dim each) ]
                                         │
                                         ▼
                   [ FAISS IndexIVFPQ Quantized Memory Bank ]
                   (nlist=100 Voronoi clusters, M=16, 8-bit PQ)
                                         │
                                         ▼
                         [ Top-k (k=5) Majority Voting ]
                                         │
                                         ▼
                        [ 16 x 16 Predicted Patch Grid ]
                                         │
                                         ▼
                 [ Full-Resolution Interpolation (1600 x 256) ]
                                         │
                                         ▼
                 [ Multi-Class Defect Segmentation Overlay ]
```

---

## 🚀 Key Highlights & Design Innovations

1. **Zero-Shot Transfer via Foundation Vision Model**:
   Utilizes self-supervised **DINOv2 ViT-B/14** patch tokens without requiring end-to-end backpropagation or heavy decoder networks.
2. **192× Memory Compression via Product Quantization**:
   High-dimensional 768-dimensional patch vectors ($3072$ bytes each) are quantized using **FAISS `IndexIVFPQ`** into $16$-byte product quantized codes, drastically lowering memory usage.
3. **Subsampled Memory Balancing**:
   Background (normal) patches undergo 5% stochastic subsampling to eliminate class imbalance in the memory bank while capturing rich surface variance.
4. **Accurate Resolution Mapping**:
   A $16 \times 16$ patch prediction grid corresponds to 14-pixel receptive fields on the normalized image and is smoothly mapped back to the original $1600 \times 256$ industrial sheet dimensions.

---

## 🛠️ Defect Classes & Visual Color Palette

| Class ID | Defect Type | Industrial Description | Inspection Color |
| :--- | :--- | :--- | :--- |
| **0** | Normal | Defect-free surface | *Transparent* |
| **1** | Pitted / Inclusion | Localized pitting or foreign particle inclusion | **Emerald Green** `(0, 220, 100)` |
| **2** | Edge Imperfection | Cracks or deformities along steel strip edges | **Amber Yellow** `(255, 200, 0)` |
| **3** | Scratch / Gouge | Linear scratches, scrapes, or roll marks | **Crimson Red** `(230, 40, 40)` |
| **4** | Patch Defect | Large-area patches, scaling, or surface stains | **Azure Blue** `(30, 140, 255)` |

---

## 📂 Repository Structure

```
Steel-Defect-Quantized-Inspector/
├── data/
│   ├── severstal/
│   │   ├── train_images/       # Extracted training images
│   │   ├── test_images/        # Extracted test images
│   │   └── train.csv           # Ground truth RLE annotations
│   ├── severstal_ivfpq.index   # Serialized FAISS IndexIVFPQ
│   └── severstal_labels.npy    # Serialized patch labels
├── notebooks/
│   └── exploration_and_demo.ipynb  # Interactive visualization notebook
├── results/                    # Generated inspection plots & overlays
├── src/
│   ├── __init__.py
│   ├── rle_utils.py            # Bidirectional Fortran-order RLE converter
│   ├── dataset.py              # PyTorch Dataset with 16x16 patch mapping
│   ├── build_index.py          # Feature extraction & FAISS IVFPQ builder
│   └── evaluate.py             # Inference, majority voting & visualization
├── tests/
│   ├── test_rle.py             # Unit tests for RLE encoding/decoding
│   ├── test_dataset.py         # Unit tests for dataset transformations
│   └── test_pipeline.py        # Unit tests for FAISS index and metrics
├── requirements.txt            # Dependency specifications
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/Steel-Defect-Quantized-Inspector.git
cd Steel-Defect-Quantized-Inspector
pip install -r requirements.txt
```

### 2. Verify Installation with Unit Tests
```bash
python -m unittest discover tests
```

---

## 📖 Usage Guide

### Step 1: Download the Severstal Dataset
Ensure your Kaggle API key is configured, then run:
```bash
kaggle datasets download -d javadseraj/severstalsteeldefect -p data/severstal --unzip
```

### Step 2: Build the Quantized Memory Bank
Extract DINOv2 patch features and construct the FAISS `IndexIVFPQ` index:
```bash
# Indexing a sample of 300 images (or leave out --max_images for entire dataset)
python -m src.build_index --img_dir data/severstal/train_images --csv_path data/severstal/train.csv --max_images 300 --batch_size 16
```

### Step 3: Run Inference & Segmentation Evaluation
Evaluate the pipeline on test/validation samples and generate visual inspection plots:
```bash
python -m src.evaluate --num_samples 10 --output_dir results
```

---

## 🔬 Mathematical & Algorithmic Formulation

### 1. Patch Token Extraction
For an input image $x \in \mathbb{R}^{3 \times 224 \times 224}$, the ViT backbone partitions the image into $N = \left(\frac{224}{14}\right)^2 = 256$ non-overlapping patches:
$$z = \text{DINOv2}(x) \in \mathbb{R}^{256 \times 768}$$

### 2. Product Quantization ($M=16, b=8$)
Each $D=768$-dimensional embedding $z_i$ is decomposed into $M=16$ orthogonal sub-vectors of dimension $d^* = 48$:
$$z_i = [u_i^1, u_i^2, \dots, u_i^{16}]$$
Each sub-vector $u_i^m$ is mapped to its nearest centroid $c_k^m \in \mathcal{C}^m$ ($2^8 = 256$ centroids per sub-space):
$$q(z_i) = [\text{idx}_1, \text{idx}_2, \dots, \text{idx}_{16}] \in \{0, \dots, 255\}^{16}$$
This yields an exact memory footprint of **16 bytes per patch**.

### 3. $k$-NN Majority Voting Classification
Given test query patch $q_i$, the top-$k$ nearest neighbors in the memory bank $\mathcal{N}_k(q_i) = \{(v_j, y_j)\}_{j=1}^k$ determine the patch class:
$$\hat{y}_i = \operatorname*{arg\,max}_{c \in \{0,1,2,3,4\}} \sum_{j=1}^k \mathbb{I}(y_j = c)$$

---

## 📄 License
MIT License. Created for robust industrial visual inspection.
