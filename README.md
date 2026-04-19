# Land Cover Classification on EuroSAT — CNN vs ResNet-50

A systematic comparison of a baseline CNN and a fine-tuned ResNet-50 for
semantic land cover classification on the EuroSAT dataset (Sentinel-2
multispectral satellite imagery). This project was completed as part of
an MSc module in Artificial Intelligence at the University of Leicester.

---

## Problem Statement

Remote sensing image classification is a core task in earth observation,
with applications in flood mapping, deforestation monitoring, urban planning
and agricultural analysis. This project investigates how architectural
choices and transfer learning affect classification performance on
multispectral satellite imagery across 10 land cover categories.

**Key research questions:**

- How does a fine-tuned pretrained ResNet-50 compare to a baseline CNN
  trained from scratch on EuroSAT?
- How does freezing depth and learning rate scheduling affect convergence
  and generalisation?
- Where do each model's inductive biases succeed or fail at the class level?

---

## Dataset

**EuroSAT** — A benchmark dataset based on Sentinel-2 satellite imagery,
covering 10 land use and land cover classes across Europe.

| Property   | Detail                                                        |
|------------|---------------------------------------------------------------|
| Classes    | 10 (AnnualCrop, Forest, Highway, Industrial, Residential ...) |
| Images     | 27,000 labelled patches                                       |
| Resolution | 64 × 64 pixels per image                                      |
| Source     | Sentinel-2 satellite (ESA) via Kaggle                         |

> Dataset: [EuroSAT on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
> Downloaded automatically via `kagglehub` in the notebook.

**Train / Val / Test split:** 19,317 / 4,139 / 4,141 (70/15/15)

---

## Models

### Baseline CNN
A custom convolutional neural network trained from scratch, serving as
the experimental control.

**Architecture:**
- 3 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool)
- Global average pooling
- Fully connected classifier (512 → 10)
- Dropout regularisation

### Fine-tuned ResNet-50
A ResNet-50 pretrained on ImageNet, with the final classification head
replaced and fine-tuned on EuroSAT.

**Fine-tuning strategy:**
- Backbone pretrained weights frozen initially; head trained first
- Full model fine-tuned end-to-end at reduced learning rate
- ImageNet normalisation applied (mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225])

---

## Results

### Overall Performance (Test Set — 4,141 samples)

| Model                 | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| Baseline CNN          | 87.56%   | 0.8808    | 0.8653 | 0.8695   |
| ResNet-50 (Fine-tuned)| 91.67%   | 0.9150    | 0.9111 | 0.9127   |

### Per-Class F1 Score

| Class                | Baseline CNN | ResNet-50 |
|----------------------|-------------|-----------|
| AnnualCrop           | 0.87        | 0.93      |
| Forest               | 0.97        | 0.98      |
| HerbaceousVegetation | 0.81        | 0.86      |
| Highway              | 0.76        | 0.84      |
| Industrial           | 0.85        | 0.89      |
| Pasture              | 0.91        | 0.93      |
| PermanentCrop        | 0.79        | 0.83      |
| Residential          | 0.87        | 0.92      |
| River                | 0.89        | 0.94      |
| SeaLake              | 0.98        | 0.99      |

### Training Summary

| Parameter            | Baseline CNN | ResNet-50         |
|----------------------|-------------|-------------------|
| Epochs               | 20          | 20                |
| Best Val Accuracy    | 86.91%      | 90.36%            |
| Optimiser            | Adam        | Adam              |
| Batch Size           | 64          | 64                |
| Random Seed          | 42          | 42                |

**Key findings:**
- ResNet-50 outperformed the baseline CNN by ~4 percentage points overall
- Transfer learning compressed the training curve significantly —
  ResNet-50 achieved 82.6% validation accuracy in epoch 1 alone,
  while the baseline CNN needed 11+ epochs to reach comparable performance
- Highway and PermanentCrop were the hardest classes for both models,
  likely due to spectral similarity with other land cover types
- SeaLake and Forest were consistently easy to classify (F1 > 0.97)
  across both architectures
- The baseline CNN showed signs of overfitting beyond epoch 14,
  while ResNet-50 maintained more stable validation performance

---

## Setup & Usage

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

The requirements.txt file contains:

- torch>=2.0.0
- torchvision>=0.15.0
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter
- Pillow
- kagglehub
- tifffile

### Running the Notebook

```bash
jupyter notebook EuroSAT_LandCover_CNN.ipynb
```

Or open directly in Google Colab — the notebook uses `kagglehub` to
download the dataset automatically. No manual data setup required.

---

## Limitations & Future Work

- RGB subset used; full 13-band multispectral Sentinel-2 data could
  improve performance on spectrally ambiguous classes like Highway
  and PermanentCrop
- Experiments run on a single T4 GPU (Google Colab); distributed
  training not explored
- Future work: U-Net for pixel-level segmentation, Vision Transformer
  (ViT) comparison, spectral augmentation strategies

---

## Academic Context

This project was completed as part of the MSc Artificial Intelligence
for Business Intelligence (with Industry) programme at the University
of Leicester, for a module on remote sensing and deep learning
applications (Assignment 2).

---

## Author

**Mohamed Rizvi Shaik Abdulla**
MSc AI for Business Intelligence, University of Leicester
[LinkedIn](https://linkedin.com/in/sam1061) ·
[GitHub](https://github.com/mdrizvi-106)
