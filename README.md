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

| Property        | Detail                        |
|-----------------|-------------------------------|
| Classes         | 10 (e.g. Forest, River, Highway, Industrial) |
| Images          | 27,000 labelled RGB patches   |
| Resolution      | 64 × 64 pixels per image      |
| Source          | Sentinel-2 satellite (ESA)    |

> Download the dataset from the 
> [official EuroSAT repository](https://github.com/phelber/EuroSAT) 
> and place it in `data/eurosat/`.

---

## Models

### Baseline CNN
A custom convolutional neural network trained from scratch, serving as the 
experimental control.

**Architecture:**
- 3 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool)
- Global average pooling
- Fully connected classifier (512 → 10)
- Dropout regularisation

### Fine-tuned ResNet-50
A ResNet-50 pretrained on ImageNet, with the final classification head 
replaced and fine-tuned on EuroSAT.

**Fine-tuning strategy:**
- Stage 1: Freeze all backbone layers, train classifier head only
- Stage 2: Unfreeze final residual blocks, train end-to-end at reduced 
  learning rate
- Learning rate scheduling: StepLR / CosineAnnealingLR

---

## Results

| Model            | Top-1 Accuracy | Precision | Recall | F1 Score |
|------------------|---------------|-----------|--------|----------|
| Baseline CNN     | XX.X%         | X.XX      | X.XX   | X.XX     |
| ResNet-50 (FT)   | XX.X%         | X.XX      | X.XX   | X.XX     |

> Fill in your actual numbers before pushing.

**Key findings:**
- Transfer learning significantly compressed the training curve, reaching 
  comparable validation accuracy in fewer epochs
- ResNet-50 showed sensitivity to spectrally unusual classes where 
  ImageNet pretraining provides limited prior knowledge
- Baseline CNN generalised more consistently on low-texture classes 
  despite lower overall accuracy

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

### Running the Experiments

**1. Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**2. Train Baseline CNN**
```bash
jupyter notebook notebooks/02_baseline_cnn.ipynb
```

**3. Fine-tune ResNet-50**
```bash
jupyter notebook notebooks/03_resnet50_finetune.ipynb
```

---

## Experimental Setup

| Parameter              | Baseline CNN | ResNet-50        |
|------------------------|-------------|------------------|
| Optimiser              | Adam        | Adam             |
| Initial LR             | 1e-3        | 1e-4 (head), 1e-5 (backbone) |
| Batch size             | 32          | 32               |
| Epochs                 | 30          | 30               |
| LR Scheduler           | StepLR      | CosineAnnealingLR|
| Data augmentation      | RandomHFlip, RandomCrop | Same  |
| Train/Val/Test split   | 70/15/15    | 70/15/15         |

---

## Visualisations

Training loss and accuracy curves, per-class confusion matrices and
sample predictions are saved to `results/figures/` after running
the evaluation notebooks.

---

## Limitations & Future Work

- EuroSAT RGB subset used; full 13-band multispectral data could improve
  performance on spectrally ambiguous classes
- Experiments were run on a single GPU; distributed training not explored
- Future work: U-Net for pixel-level segmentation, Vision Transformer (ViT)
  comparison, data augmentation with spectral jitter

---

## Academic Context

This project was completed as part of the MSc Artificial Intelligence for
Business Intelligence (with Industry) programme at the University of
Leicester, for a module on remote sensing and deep learning applications.

---

## Author

**Mohamed Rizvi Shaik Abdulla**
MSc AI for Business Intelligence, University of Leicester
[LinkedIn](https://linkedin.com/in/sam1061) ·
[GitHub](https://github.com/mdrizvi-106)
