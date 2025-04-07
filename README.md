# Weakly Supervised Barrett's Dysplasia Grading

This repository provides code for predicting dysplasia grade from whole-slide images (WSIs) of Barrett's esophagus biopsies using weakly supervised learning. This approach is based on Multiple Instance Learning (MIL) to perform slide-level classification based on patch-level features extracted from WSIs.

## Main components

- **`feature_extraction.py`**  
  Extracts patch-level features from WSIs using a pre-trained deep learning model: 
  - [Virchow2](https://huggingface.co/NKI-AI/virchow2-vit-small)
  - [Conch](https://huggingface.co/NKI-AI/conch-vit-small)
  - Image tiling is performed using [DLUP](https://github.com/NKI-AI/dlup).

- **`train.py`**  
  Trains a MIL-based model to aggregate extracted features and predict an overall dysplasia grade for each slide. 
