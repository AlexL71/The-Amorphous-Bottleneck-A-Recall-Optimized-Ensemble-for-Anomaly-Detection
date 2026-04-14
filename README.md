# The Amorphous Bottleneck: A Recall-Optimized Ensemble for Anomaly Detection

## Abstract

Traditional object detection frameworks underperform on ill-defined anomalies lacking clear geometric boundaries. This research identifies this failure as **The Amorphous Bottleneck** and proposes a data-centric, recall-optimized ensemble framework to resolve it.

Using road defect detection as a case study, we train 19 specialist Mask R-CNN models (each with a unique augmentation), audit them via a Hit/Miss tracking matrix, fuse predictions with **Weighted Boxes Fusion (WBF)**, and select the optimal 6-model ensemble through a **Greedy Forward Selection (GFS)** algorithm governed by a custom Safety Score.

### Key Results (Test Set)

|Metric|Baseline|Ensemble|
|-|-|-|
|mAP@50:95|0.5420|**0.5598**|
|AP@50|0.7958|**0.8033**|
|Recall (conf > 0.50)|0.8317|**0.9089**|
|Rescues / Regressions|??**135 / 5**|
|Ghost Detections|??168|

The ensemble shifts the optimal F1 confidence threshold from 0.75 to 0.90, producing highly decisive predictions. The 168 Ghost detections provide empirical evidence of the **Precision Paradox**: standard precision metrics penalize models for detecting valid defects absent from the ground truth.

## Repository Structure

```
amorphous-bottleneck/
?œâ??€ configs/
??  ?”â??€ mask-rcnn\\\\\\\_r50\\\\\\\_fpn\\\\\\\_1x\\\\\\\_coco.py       # MMDetection base config
?œâ??€ notebooks/
??  ?œâ??€ 00\\\\\\\_model\\\\\\\_training.ipynb             # 20 experiment training configs (code only)
??  ?œâ??€ 01\\\\\\\_dataset\\\\\\\_analysis.ipynb           # EDA: scale + Weber contrast analysis
??  ?œâ??€ 02\\\\\\\_model\\\\\\\_audit.ipynb                # Val inference + Hit/Miss leaderboard
??  ?œâ??€ 03\\\\\\\_team\\\\\\\_selection.ipynb             # GFS algorithm + team export
??  ?œâ??€ 04\\\\\\\_ensemble\\\\\\\_inference.ipynb         # Test set WBF fusion engine
??  ?œâ??€ 05\\\\\\\_final\\\\\\\_evaluation.ipynb           # COCO metrics + forensic breakdown
??  ?”â??€ 06\\\\\\\_visual\\\\\\\_audit.ipynb              # Rescue/Regression/Ghost visualization
?œâ??€ outputs/
??  ?œâ??€ val\\\\\\\_results\\\\\\\_\\\\\\\*.json                  # Validation predictions (20 models)
??  ?œâ??€ test\\\\\\\_results\\\\\\\_\\\\\\\*.json                 # Test predictions (6 team models + ensemble)
??  ?œâ??€ test\\\\\\\_results\\\\\\\_ensemble\\\\\\\_grand.json    # Final fused ensemble output
??  ?œâ??€ final\\\\\\\_team\\\\\\\_config.json              # GFS-selected team + WBF parameters
??  ?”â??€ val\\\\\\\_leaderboard.csv                 # Model ranking by Absolute Score
?œâ??€ figures/
??  ?œâ??€ eda/                                # Figures 1-3 (area, contrast, demo)
??  ?œâ??€ results/                            # Figures 7-8 (PR curves, F1 curves)
??  ?”â??€ visual\\\\\\\_audit/                       # Figures 9-11 (Rescues, Regressions, Ghosts)
??      ?œâ??€ RESCUES/                        # 116 annotated rescue images
??      ?œâ??€ REGRESSIONS/                    # 5 annotated regression images
??      ?œâ??€ BOTH\\\\\\\_MISSED/                    # 115 annotated missed images
??      ?”â??€ GHOSTS/                         # 114 annotated ghost images
?œâ??€ data/
??  ?”â??€ sample/                             # Small sample for code verification
??      ?œâ??€ sample\\\\\\\_annotations.json
??      ?”â??€ images/                         # 10 sample images
?œâ??€ docs/
??  ?”â??€ ABSTRACT.pdf                          # Full thesis document will be uploaded 2026/07
?œâ??€ requirements.txt
?”â??€ README.md
```

## Pipeline Overview



!\[Methodology Framework](figures/methodology\_framework.png)

&#x20;

## Data \& Model Weights

The training dataset and model weights are **not included** in this repository. The data was collected as part of a government-funded research project:

> \\\\\\\*\\\\\\\*Cloud-sourcing-based Mobility Support\\\\\\\*\\\\\\\*   
> Period: March 2025 ??December 2025  
> Organization: Big Data Mining Lab, Korea University  



Due to institutional ownership restrictions, only a small data sample is provided for code verification purposes.

### Expected data structure (for running notebooks):

```
data/
?œâ??€ train/
??  ?œâ??€ images/          # 1,769 images (1920Ã—648)
??  ?”â??€ train\\\\\\\_annotations.json
?œâ??€ val/
??  ?œâ??€ images/          # 589 images
??  ?”â??€ val\\\\\\\_annotations.json
?”â??€ test/
    ?œâ??€ images/          # 591 images
    ?”â??€ test\\\\\\\_annotations.json
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

> \\\\\\\*\\\\\\\*Note:\\\\\\\*\\\\\\\* MMDetection requires separate installation. See \\\\\\\[MMDetection docs](https://mmdetection.readthedocs.io/en/latest/get\\\\\\\_started.html).



## Hardware

All experiments were conducted on:

* **GPU:** NVIDIA GeForce RTX 4080
* **Framework:** PyTorch 2.0.1 + CUDA 11.8
* **OS:** Windows 11



## License

This project is released for academic and research purposes. The source code is provided as-is for reproducibility. The training data and model weights remain property of the originating research project.

