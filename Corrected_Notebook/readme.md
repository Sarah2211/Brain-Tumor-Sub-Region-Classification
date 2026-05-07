# README

This folder contains the corrected Jupyter notebooks for the BraTS2020 brain tumor classification project.

# Files

## Demo.ipynb
This is a small runnable demo notebook. It uses a tiny real subset of the BraTS2020 dataset uploaded through Google Drive.

The demo notebook:
- downloads and extracts the tiny BraTS dataset
- loads real MRI files
- uses FLAIR, T1, T1CE, and T2 modalities
- uses segmentation masks to create tumor-region labels
- runs the data through the model pipeline
- shows metrics and graphs

The demo dataset is only for checking that the pipeline works. It is not used for final accuracy.

---

## EfficientNet_BraTS2020.ipynb
This notebook trains and evaluates an EfficientNet-based CNN model for multi-label brain tumor region classification.

---

## ViT_BraTS2020.ipynb
This notebook trains and evaluates a Vision Transformer model for BraTS2020 MRI classification.

---

## SwinT_BraTS2020.ipynb
This notebook trains and evaluates a Swin Transformer model for brain tumor classification.

---

## SwinT_MIL_BraTS2020.ipynb
This notebook trains and evaluates a Swin Transformer with Multiple Instance Learning model. It uses slice-level features and combines them for patient-level prediction.

# Dataset Used

The main notebooks use the BraTS2020 dataset.

The demo notebook uses a tiny real subset of the BraTS2020 dataset. It includes only a few patient folders so the notebook can run quickly.

MRI modalities used:
- FLAIR
- T1
- T1CE
- T2

Tumor labels predicted:
- NCR/NET
- ED
- ET