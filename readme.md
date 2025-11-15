# LLD-YOLO: A Lightweight Low-Light Object Detector Trained on ExDark

This repository contains the research implementation of **LLD-YOLO**, a lightweight YOLO-based architecture optimized for **low-light object detection** on the **ExDark** dataset. The project includes modular model components, training/evaluation pipelines, improved architectures, confusion matrices, and publication-ready results.

---

## Repository Structure
- **config.py** — Experiment configuration  
- **run.py / run_improved.py** — Training pipelines  
- **compare_models.py** — Model comparison script  
- **paper_table.py** — Generates research tables  
- **evaluate.py** — Evaluation logic  
- **data/** — Dataset loader, annotation parser, augmentations  
- **models/** — Backbones, attention modules, detection heads  
- **train/** — Losses, metrics, training engine  
- **weights/** — Pretrained weights  
- **results/** — JSON logs and visualizations  

---

## ExDark Dataset Overview

The **ExDark (Extreme Low-Light Dataset)** contains images captured exclusively in low-light conditions.

### Key Characteristics
- 7,363 low-light images  
- 12 classes: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table  
- Real-world night and low-illumination settings  
- High noise, low contrast, strong shadows  

### Dataset Format
- Images: `data/ExDark/images/`  
- YOLO labels: `data/ExDark/labels/`  

### Preprocessing Applied
- Gamma correction  
- CLAHE  
- Normalization  
- Rotation/scale augmentation  
- Noise-aware augmentations  

---

## Model Architecture

### Original LLD-YOLO
- Lightweight backbone  
- Basic feature fusion  
- Standard YOLO detection head  

### Improved LLD-YOLO
- Low-Light Enhancement Module (LL-EM)  
- Enhanced backbone (improved convolution blocks)  
- Optional DBB / Ghost / ELAN modules  
- Improved CIoU/EIoU loss  

### Design Goals
- Better low-light feature extraction  
- Faster inference  
- Higher stability during training  

---

## Results on ExDark

### Performance Metrics

| Model                 | mAP@50 | mAP@50-95 | Precision | Recall | F1-score | FPS  | Latency (ms) |
|-----------------------|--------|-----------|-----------|--------|----------|------|---------------|
| Original LLD-YOLO     | 0.423  | 0.219     | 0.61      | 0.54   | 0.57     | 42.7 | 23.4          |
| Improved LLD-YOLO     | 0.487  | 0.261     | 0.67      | 0.59   | 0.62     | 45.1 | 21.3          |

The improved model achieves a **+6.4% gain in mAP@50** and a **+4.2% gain in mAP@50–95**.

---

## Visualizations

### Precision–Recall Curve
![PR Curve](https://github.com/rishavafk/LLD_Research/raw/main/results/figures/pr_curve.png)

### Loss Curves
![Loss Curves](https://github.com/rishavafk/LLD_Research/raw/main/results/figures/loss_curves.png)

### Metrics Barplot
![Metrics Barplot](https://github.com/rishavafk/LLD_Research/raw/main/results/figures/metrics_barplot.png)

### Confusion Matrices

#### Original Model
![Confusion Matrix - Original](https://github.com/rishavafk/LLD_Research/raw/main/results/figures/confusion_original.png)

#### Improved Model
![Confusion Matrix - Improved](https://github.com/rishavafk/LLD_Research/raw/main/results/figures/confusion_improved.png)

---

## Research Findings

### 1. Accuracy & Detection Quality
Improvements due to:
- Stronger low-light feature extraction  
- Reduced noise sensitivity  
- Better bounding box regression (CIoU/EIoU)  

### 2. Training Stability
- Faster convergence  
- Lower oscillation  
- Smooth loss curves  

### 3. Real-Time Performance
- FPS: 42.7 → 45.1  
- Latency: 23.4 ms → 21.3 ms  

---

## Research Summary

LLD-YOLO combines efficient backbone design with illumination-aware enhancement modules to significantly improve performance on low-light datasets like ExDark. The improved architecture delivers higher accuracy, better convergence, and faster inference while remaining lightweight and deployment-ready.

---

## Citation

Use the following citation for the ExDark dataset:

