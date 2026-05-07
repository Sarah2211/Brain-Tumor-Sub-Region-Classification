# 🧠 Brain Tumor Sub-Region Classification on BraTS2020

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=flat-square&logo=googlecolab)


Patient-level multi-label binary classification of brain tumor sub-regions, benchmarking four deep learning architectures — two CNNs and two Transformers — on the BraTS2020 dataset. Each model processes a full sequence of multi-modal MRI slices and predicts the presence or absence of clinically significant tumor regions.

---

## 🎯 Problem Statement

Brain tumors, particularly gliomas, are characterized by distinct sub-regions that differ in biology and clinical significance. Accurately identifying which sub-regions are present in a patient's scan supports diagnosis and treatment planning.

Given a patient's MRI scan sequence across 4 modalities **(T1, T1ce, T2, FLAIR)**, the task is to predict a **3-dimensional binary label**:

| Sub-Region | Label | Clinical Meaning |
|------------|-------|-----------------|
| Necrotic & Non-Enhancing Tumor Core (NCR) | 1 | Dead/inactive tumor tissue |
| Peritumoral Edema (ED) | 2 | Swelling around the tumor |
| Enhancing Tumor (ET) | 4 | Active, contrast-enhancing region |

This is a **multi-label classification** problem — a patient can have any combination of the three sub-regions present.

---

## 🗂️ Dataset

**BraTS2020 — Brain Tumor Segmentation Challenge 2020**

- **Patients**: 369 glioma patients
- **Slices**: 155 MRI slices per patient (57,195 total)
- **Input shape per slice**: (240, 240, 4) resized to (224, 224, 4)
- **Labels**: Patient-level binary vector [NCR, ED, ET] derived from pixel-level masks
- **Source**: [Kaggle — awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

> Splits are done at the **patient level** (70/15/15) to prevent data leakage across slices from the same patient.

---

## 🏗️ Models & Architecture

All models share the same pipeline structure:

    MRI Slices (per patient)
          ↓
    Slice-level Feature Extraction (shared backbone)
          ↓
    Patient-level Aggregation
          ↓
    Classification Head → sigmoid → [NCR, ED, ET]

Four architectures are benchmarked, varying the backbone and aggregation strategy:

### 1. EfficientNet-B3 + Mean Pooling
- Pretrained EfficientNet-B3 encodes each slice into a **1536-dim** feature vector
- Mean pooling across all slices produces a single patient-level representation
- Strong CNN baseline using compound scaling

### 2. ViT-B/16 + Mean Pooling
- Each slice is tokenized into **16×16 patches** and encoded by a pretrained ViT
- **768-dim** CLS token per slice, aggregated via mean pooling
- Captures global within-slice relationships via self-attention

### 3. Swin Transformer + Temporal Transformer
- Swin-T encodes each slice using **hierarchical shifted-window attention**
- Slice embeddings + learned positional encodings fed into a **4-layer Temporal Transformer**
- Models inter-slice dependencies to capture 3D volumetric context

### 4. Swin Transformer + MIL Attention Pooling
- Same Swin-T backbone as above
- Slice aggregation via **Multiple Instance Learning (MIL)** attention
- Learns to assign higher weights to diagnostically relevant (tumor-containing) slices

---

## ⚙️ Training Setup

| Parameter | Value |
|-----------|-------|
| Loss Function | BCEWithLogitsLoss with class weights |
| Optimizer | AdamW (differential LR) |
| Backbone LR | 1e-4 |
| Classifier LR | 1e-3 |
| LR Scheduler | Cosine Annealing |
| Max Epochs | 20 |
| Early Stopping | Patience = 5 |
| Dropout | 0.3 |
| Mixed Precision | AMP (fp16) |
| Platform | Google Colab (T4 GPU) |

**Regularization**: Dropout in classification head, differential learning rates to preserve pretrained features, cosine LR scheduling, early stopping on validation loss, and class-weighted loss for label imbalance.

---

## 📊 Evaluation Metrics

Every model is evaluated on a held-out test set and outputs a standardized results.json containing:

- ✅ **Classification Report** — Precision, Recall, F1 per class
- ✅ **Confusion Matrix** — per tumor sub-region
- ✅ **AUROC** — per class and macro average
- ✅ **Log Loss** — per class and mean

Cross-model comparison and visualizations are in visualization/comparison.ipynb.

---

## 📚 References

1. [EfficientNet](https://arxiv.org/pdf/1905.11946) — Tan & Le, 2019
2. [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020
3. [Swin Transformer](https://arxiv.org/abs/2103.14030) — Liu et al., 2021
4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
5. [MIL Attention Pooling](https://arxiv.org/abs/1802.04712) — Ilse et al., 2018
6. [BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
