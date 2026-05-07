# Brain Tumor Sub-Region Classification on BraTS 2020

> **CS 518 – Group 3** | Sarah Zahir Syeda · Gauthami Ghadiyaram · Sudha Sree Yerramsetty · Shriraksha Srinivas

Patient-level multi-label binary classification of glioma sub-regions from multi-modal MRI using the BraTS 2020 dataset. We implement and compare four deep learning pipelines, each pairing a different backbone with a different slice aggregation strategy.

[BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) &nbsp;|&nbsp; [Demo Notebook](./Demo.ipynb)

---

## Problem

- Given a patient's full MRI volume across 4 modalities (T1, T1ce, T2, FLAIR), predict which tumor sub-regions are present
- Output is a 3-dimensional binary label vector: **[NCR, ED, ET]**
- Labels are not mutually exclusive, any combination can be present
- Each patient has 155 axial slices, tumor tissue typically appears in only 10–40 of them

| Sub-Region | Meaning |
|---|---|
| NCR/ Necrotic Core | Dead/inactive tumor tissue |
| ED/ Peritumoral Edema | Swelling around the tumor |
| ET/ Enhancing Tumor | Actively growing, contrast-enhancing region |

---

## Dataset

- **BraTS2020 Segmentation Dataset** - 369 glioma patients (293 HGG, 76 LGG)
- 155 MRI slices per patient across T1, T1ce, T2, FLAIR modalities
- Raw resolution: 240 × 240 × 4 per slice
- Patient-level labels derived from pixel-level segmentation masks
- NCR present in 281 patients (76.2%), ED in 369 (100%), ET in 261 (70.7%)
- Split: 70% train / 15% val / 15% test done at the patient level to prevent data leakage

---

## Preprocessing

- Loaded HDF5 files, transposed to channel-first format (4, 240, 240)
- Per-channel min-max normalization to [0, 1]
- Bilinearly resized to 224 × 224 to match pretrained backbone input
- Training augmentation: random horizontal and vertical flips (p = 0.5)
- `BCEWithLogitsLoss` with per-label positive weights to handle class imbalance
- Fixed decision threshold of 0.5 at inference

---

## Models

All four models follow the same two-stage design: **slice encoder** (per slice) → **aggregation** (155 slices → 1 patient vector) → **classification head** → sigmoid → [NCR, ED, ET].

### 1. EfficientNet-B3 + Mean Pooling
- EfficientNet-B3 pretrained on ImageNet, input layer adapted from 3 → 4 channels
- Each slice encoded to a 1536-dim embedding
- All 155 embeddings averaged into one patient vector
- Head: `LayerNorm → Dropout(0.3) → Linear(1536, 128) → GELU → Dropout(0.15) → Linear(128, 3)`
- AdamW, backbone LR = 1e-4, head LR = 1e-3, cosine annealing, 20 epochs

### 2. ViT-B/16 + Mean Pooling
- ViT-B/16 pretrained on ImageNet, 4-channel input via `in_chans=4` in timm
- Each slice divided into 196 patches of 16 × 16, processed through 12 self-attention layers
- 768-dim CLS token per slice; slices processed in chunks of 16 due to memory constraints
- All 155 CLS tokens averaged into one patient vector
- Head: `LayerNorm → Dropout(0.3) → Linear(768, 256) → GELU → Dropout(0.3) → Linear(256, 128) → GELU → Dropout(0.15) → Linear(128, 3)`
- AdamW, backbone LR = 1e-5, head LR = 1e-4, cosine annealing, 30 epochs

### 3. Swin-T + Temporal Transformer
- Swin Transformer (Swin-T) with local shifted-window attention, 4-channel input
- Each slice encoded to a 768-dim embedding
- 155 ordered slice embeddings passed through a Temporal Transformer (2 layers, 8 heads, FFN dim = 3072) with learned positional encodings
- Mean of the 155 output tokens taken as the patient representation
- Head: `Linear → GELU → Dropout(0.3) → Linear(3)`
- AdamW, Swin LR = 1e-5, temporal transformer + head LR = 1e-4, batch size = 4

### 4. Swin-T + MIL Attention Pooling
- Same Swin-T backbone as Model 3
- MIL attention module learns a scalar importance weight per slice: `A = softmax(W₂ · tanh(W₁ · Hᵀ))`, patient vector = weighted sum of slice embeddings
- Focuses on tumor-bearing slices, suppresses uninformative background slices
- Head: `Linear(768, 3)` with sigmoid
- AdamW, Swin LR = 1e-5, MIL + head LR = 1e-4, early stopping (patience = 7)

---

## Training Setup

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Weight Decay | 1e-4 |
| Mixed Precision | FP16 (AMP) |
| Gradient Clipping | 1.0 |
| Loss Function | BCEWithLogitsLoss with class weights |
| Hardware | NVIDIA A100-SXM4-80GB |
| Framework | PyTorch 2.1, CUDA 12.8, timm 0.9.x |

---

## Results

| Model | Accuracy | Macro AUROC | Log Loss | Macro F1 |
|---|---|---|---|---|
| EfficientNet-B3 + Mean Pooling | 98.00% | 0.4434 | **0.1169** | 0.9908 |
| ViT-B/16 + Mean Pooling | 97.00% | 0.2893 | 0.6600 | 0.9908 |
| **Swin-T + Temporal Transformer** | **98.21%** | **0.7138** | 0.6412 | 0.9908 |
| Swin-T + MIL Attention | **98.21%** | 0.6981 | 0.2984 | 0.9908 |

- We see **AUROC and log loss** show true model performance
- Swin-T + Temporal Transformer achieves the highest AUROC (0.714)
- Swin-T + MIL achieves the best log loss (0.298) and best overall balance between discrimination and calibration

---

## Repository Structure

```
Brain-Tumor-Sub-Region-Classification/
│
├── README.md
└── Model_Notebooks/
    ├── EfficientNet_B3_MeanPool.ipynb
    ├── ViT_B16_MeanPool.ipynb
    ├── SwinT_TemporalTransformer.ipynb
    ├── SwinT_MIL_Attention.ipynb
    └── Demo.ipynb

```

---

## How to Run

**Quick demo (no full dataset needed):**
- Open `Demo.ipynb` in Google Colab
- Runs all four models on a 20-patient mini-subset in under 15 minutes on free-tier GPU

**Full training:**
1. Download BraTS 2020 from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
2. Open the relevant notebook from `Model_Notebooks/`
3. Update the dataset path at the top of the notebook
4. Run on a GPU with ≥16 GB VRAM

---

## References

1. Tan & Le — [EfficientNet](https://arxiv.org/abs/1905.11946), ICML 2019
2. Dosovitskiy et al. — [ViT](https://arxiv.org/abs/2010.11929), ICLR 2021
3. Liu et al. — [Swin Transformer](https://arxiv.org/abs/2103.14030), ICCV 2021
4. Ilse et al. — [MIL Attention Pooling](https://arxiv.org/abs/1802.04712), ICML 2018
5. Loshchilov & Hutter — [AdamW](https://arxiv.org/abs/1711.05101), ICLR 2019
6. Menze et al. — [BraTS Benchmark](https://ieeexplore.ieee.org/document/6975210), IEEE TMI 2015
