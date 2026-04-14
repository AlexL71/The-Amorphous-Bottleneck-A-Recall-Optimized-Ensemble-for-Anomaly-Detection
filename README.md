# The Amorphous Bottleneck: A Recall-Optimized Ensemble for Anomaly Detection

> **Master's Thesis** — Dadaboev Abdurakhmon  
> Department of Big Data Science, Korea University  
> Advisor: Professor Seohoon Jin | August 2026

## Abstract

Traditional object detection frameworks underperform on ill-defined anomalies lacking clear geometric boundaries. This research identifies this failure as **The Amorphous Bottleneck** and proposes a data-centric, recall-optimized ensemble framework to resolve it.

Using road defect detection as a case study, we train 19 specialist Mask R-CNN models (each with a unique augmentation), audit them via a Hit/Miss tracking matrix, fuse predictions with **Weighted Boxes Fusion (WBF)**, and select the optimal 6-model ensemble through a **Greedy Forward Selection (GFS)** algorithm governed by a custom Safety Score.

### Key Results (Test Set)

| Metric | Baseline | Ensemble |
|--------|----------|----------|
| mAP@50:95 | 0.5420 | **0.5598** |
| AP@50 | 0.7958 | **0.8033** |
| Recall (conf > 0.50) | 0.8317 | **0.9089** |
| Rescues / Regressions | — | **135 / 5** |
| Ghost Detections | — | 168 |

The ensemble shifts the optimal F1 confidence threshold from 0.75 to 0.90, producing highly decisive predictions. The 168 Ghost detections provide empirical evidence of the **Precision Paradox**: standard precision metrics penalize models for detecting valid defects absent from the ground truth.

## Repository Structure

```
amorphous-bottleneck/
├── configs/
│   └── mask-rcnn_r50_fpn_1x_coco.py       # MMDetection base config
├── notebooks/
│   ├── 00_model_training.ipynb             # 20 experiment training configs (code only)
│   ├── 01_dataset_analysis.ipynb           # EDA: scale + Weber contrast analysis
│   ├── 02_model_audit.ipynb                # Val inference + Hit/Miss leaderboard
│   ├── 03_team_selection.ipynb             # GFS algorithm + team export
│   ├── 04_ensemble_inference.ipynb         # Test set WBF fusion engine
│   ├── 05_final_evaluation.ipynb           # COCO metrics + forensic breakdown
│   └── 06_visual_audit.ipynb              # Rescue/Regression/Ghost visualization
├── outputs/
│   ├── val_results_*.json                  # Validation predictions (20 models)
│   ├── test_results_*.json                 # Test predictions (6 team models + ensemble)
│   ├── test_results_ensemble_grand.json    # Final fused ensemble output
│   ├── final_team_config.json              # GFS-selected team + WBF parameters
│   └── val_leaderboard.csv                 # Model ranking by Absolute Score
├── figures/
│   ├── eda/                                # Figures 1-3 (area, contrast, demo)
│   ├── results/                            # Figures 7-8 (PR curves, F1 curves)
│   └── visual_audit/                       # Figures 9-11 (Rescues, Regressions, Ghosts)
│       ├── RESCUES/                        # 116 annotated rescue images
│       ├── REGRESSIONS/                    # 5 annotated regression images
│       ├── BOTH_MISSED/                    # 115 annotated missed images
│       └── GHOSTS/                         # 114 annotated ghost images
├── data/
│   └── sample/                             # Small sample for code verification
│       ├── sample_annotations.json
│       └── images/                         # 10 sample images
├── docs/
│   └── thesis.pdf                          # Full thesis document
├── requirements.txt
└── README.md
```

## Pipeline Overview

```
Phase I: Specialist Generation
  └─ 1 baseline + 19 augmented Mask R-CNN (ResNet-50-FPN) models
       └─ Spatial (8) + Pixel (8) + Occlusion (3) augmentations

Phase II: Audit & Selection (on validation set)
  └─ Hit/Miss Matrix → Safety Score (S = TTP − 0.1×TFP) → GFS Algorithm
       └─ 30 WBF parameter combos per step → 6-model team selected

Phase III: Fusion & Deployment (on test set)
  └─ WBF fusion (IoU=0.55, Conf=0.90) → 3,018 final detections
       └─ Forensic audit: 135 Rescues, 5 Regressions, 168 Ghosts
```

## Data & Model Weights

The training dataset and model weights are **not included** in this repository. The data was collected as part of a government-funded research project:

> **Cloud-sourcing-based Mobility Support** (Project No. R2320973)  
> Period: March 2025 – December 2025  
> Organization: Big Data Mining Lab, Korea University  
> Industry Partner: AI Works Co., Ltd.  
> Certification: Model Test Report No. TWR-202512-A-0072  

Due to institutional ownership restrictions, only a small data sample is provided for code verification purposes.

### Expected data structure (for running notebooks):
```
data/
├── train/
│   ├── images/          # 1,769 images (1920×648)
│   └── train_annotations.json
├── val/
│   ├── images/          # 589 images
│   └── val_annotations.json
└── test/
    ├── images/          # 591 images
    └── test_annotations.json
```

**Dataset statistics:** 2,949 images | 9,028 annotations | 2 classes (ac: alligator cracks, lc: longitudinal cracks)

## Requirements

```
# Core
python>=3.10
torch>=2.0
mmdetection>=3.0
mmengine>=0.10

# Data processing
pycocotools
albumentations
opencv-python

# Analysis
numpy
pandas
matplotlib
seaborn
scipy
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

> **Note:** MMDetection requires separate installation. See [MMDetection docs](https://mmdetection.readthedocs.io/en/latest/get_started.html).

## Hardware

All experiments were conducted on:
- **GPU:** NVIDIA GeForce RTX 4080
- **Framework:** PyTorch 2.0.1 + CUDA 11.8
- **OS:** Windows 11

## Citation

```bibtex
@mastersthesis{dadaboev2026amorphous,
  title   = {The Amorphous Bottleneck: A Recall-Optimized Ensemble 
             for Anomaly Detection},
  author  = {Dadaboev, Abdurakhmon},
  school  = {Korea University},
  year    = {2026},
  type    = {Master's Thesis}
}
```

## License

This project is released for academic and research purposes. The source code is provided as-is for reproducibility. The training data and model weights remain property of the originating research project.
