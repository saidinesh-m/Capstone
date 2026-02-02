# Deep Learning–Based Cardiovascular Disorder Detection Using Chest X-Ray Images

## Project Objective
Develop a fully functional deep learning system that automatically analyzes chest X-ray images to detect cardiovascular abnormalities, with a primary focus on cardiomegaly and indirect cardiac dysfunction indicators.

## Problem Statement
Cardiovascular diseases are the leading cause of global mortality. Early detection is critical but manual interpretation of chest X-rays is time-consuming. This project uses DenseNet-121 to provide automated screening.

## Features
- **Preprocessing**: CLAHE, Denoising, Normalization.
- **Model**: DenseNet-121 (Transfer Learning from ImageNet).
- **Explainability**: Grad-CAM heatmaps for interpretability.
- **Metrics**: AUC, F1-Score, Sensitivity.

## Folder Structure
- `data/`: Dataset handling.
- `preprocessing/`: Image transformations.
- `models/`: CNN definitions.
- `training/`: Training loops.
- `evaluation/`: Metrics and Testing.
- `explainability/`: Grad-CAM implementation.
- `utils/`: Helpers.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your dataset path in `main.py` or the config file.
3. Run training:
   ```bash
   python main.py --mode train
   ```
