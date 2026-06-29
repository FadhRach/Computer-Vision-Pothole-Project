# Pothole Detection - Computer Vision Final Project

**BINUS University - COMP7116001 - Computer Vision**

A comparative study of pothole detection pipelines using traditional computer vision — from unsupervised methods to supervised gradient boosting, without deep learning. Two fully deployed Streamlit applications allow real-time road pothole detection from images and video.

---

## Group 2 - Team Members

| No  | Name                                  | Student ID |
| --- | ------------------------------------- | ---------- |
| 1   | Dian Rakhmawati Lestari               | 2802539085 |
| 2   | Fadhlan Nur Rachman                   | 2802491690 |
| 3   | Matthew Ken Susanto                   | 2802407736 |
| 4   | Nasauramecca Nour Haqqanshah Shodiqin | 2802541921 |
| 5   | Nicholas                              | 2802424326 |

---

## Links & Resources

| Resource                   | Description                                 | Link                                                        |
| -------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| **Live App Pipeline 1**    | Random Forest + SLIC (Streamlit Cloud)      | https://computer-vision-pothole-project.streamlit.app/      |
| **Live App Pipeline 2**    | LightGBM + 36 Features (Streamlit Cloud)    | https://pothole-detection-cv.streamlit.app                  |
| **Live App Pipeline 2**    | LightGBM + 36 Features (HuggingFace Spaces) | https://huggingface.co/spaces/ddrlvee/pothole-detection     |
| **Source Code Pipeline 1** | GitHub Repository                           | https://github.com/FadhRach/Computer-Vision-Pothole-Project |
| **Source Code Pipeline 2** | GitHub Repository                           | https://github.com/ddrlve/pothole-detection                 |
| **Demo Video**             | YouTube (< 5 minutes)                       | https://youtu.be/JAk7gb7jytc?feature=shared                 |
| **Presentation (PPT)**     | Canva Slides                                | https://canva.link/nfw5gd1jus2nqk8                          |
| **Dataset ARA 7.0**        | Primary training dataset                    | ARA 7.0 Road Surface Competition                            |
| **Dataset RDD2022 India**  | Hard-negative augmentation                  | https://github.com/sekilab/RoadDamageDetector               |

---

## Project Overview

This project investigates automated pothole detection using exclusively traditional (non-deep-learning) computer vision. We systematically explored four approaches, adaptive thresholding, K-Means clustering, Random Forest with SLIC superpixels, and gradient boosting with pixel-level features, comparing their segmentation performance on the ARA 7.0 Road Surface dataset.

### Key Findings

| Pipeline       | Method                                   | mIoU      | Pixel Acc | Macro F1  |
| -------------- | ---------------------------------------- | --------- | --------- | --------- |
| Baseline       | Adaptive Threshold                       | 0.403     | **66.7%** | —         |
| Advanced       | K-Means Clustering                       | 0.444     | **75%**   | —         |
| **Pipeline 1** | **Random Forest + SLIC (9 features)**    | **0.518** | **86.3%** | **0.553** |
| **Pipeline 2** | **LightGBM + Pixel-Level (36 features)** | **0.592** | **87.4%** | **0.827** |

**LightGBM (Pipeline 2) is the best-performing model**, achieving higher mIoU, pixel accuracy, and macro F1 than all other approaches, including Random Forest (Pipeline 1).

---

## Pipeline 1 - Random Forest + SLIC

**Repository**: https://github.com/FadhRach/Computer-Vision-Pothole-Project

### How It Works

```
Input Image (BGR)
    │
    ├─ Road Segmentation     → remove sky (Hue 85–140) & vegetation (Hue 20–85)
    ├─ Preprocessing         → illumination normalization + CLAHE + bilateral filter
    ├─ Feature Extraction    → 9 features per pixel (BGR×3, HSV×3, Gradient, Blackhat, LBP)
    ├─ SLIC Superpixels      → ~250 superpixels (compactness=10, sigma=1.0)
    ├─ Feature Aggregation   → average features per superpixel
    ├─ Random Forest (200 trees, class_weight='balanced')
    └─ Postprocessing        → morphology + area filter
```

**9 Features:** BGR (Blue, Green, Red) · HSV (Hue, Saturation, Value) · Gradient Magnitude · Blackhat · Local Binary Pattern (LBP)

### Key Parameters

| Parameter                    | Value         |
| ---------------------------- | ------------- |
| SLIC n_segments              | 250           |
| SLIC compactness             | 10.0          |
| SLIC sigma                   | 1.0           |
| RF n_estimators              | 200           |
| RF class_weight              | balanced      |
| Superpixel pothole threshold | 40%           |
| Work scale                   | 0.5×          |
| CLAHE clip limit             | 2.5, tile 8×8 |

### How to Run (Pipeline 1)

```bash
# Clone the repository
git clone https://github.com/FadhRach/Computer-Vision-Pothole-Project
cd Computer-Vision-Pothole-Project

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run app/app.py
```

App opens at `http://localhost:8501`. Stop with `Ctrl+C`.

### Folder Structure (Pipeline 1)

```
Computer-Vision-Pothole-Project/
├── app/
│   └── app.py                        # Streamlit app (Detection + Model Info pages)
├── notebooks/
│   ├── main_training.ipynb           # Training & evaluation (RF, KMeans, AdaptiveThreshold)
│   └── experimentensemble_training.ipynb  # Boosting experiment on 9 features
├── model/
│   ├── model_rf_9ch.joblib           # Random Forest model (~51 MB)
│   └── model_boost_9ch.joblib        # CatBoost experiment model
├── Dataset/
│   ├── train/images/                 # 498 training images
│   ├── train/mask/                   # 498 ground truth masks
│   └── test/images/                  # 295 test images
├── result_csv/                       # Superpixel features + per-model metrics CSV
├── docs/                             # Report, presentation, requirements
├── requirements.txt
└── README.md
```

---

## Pipeline 2 - LightGBM + 36 Pixel-Level Features

**Repository**: https://github.com/ddrlve/pothole-detection

### How It Works

```
Input Image (BGR)
    │
    ├─ Preprocessing         → CLAHE + illumination normalization + bilateral filter + blackhat
    ├─ Feature Extraction    → 36 features per pixel across 9 groups
    ├─ SLIC Smoothing        → n=250, compactness=10 (label voting for spatial consistency)
    ├─ LightGBM Classifier   → pixel-level binary classification
    └─ Postprocessing        → threshold=0.80, min_area=1200px, morphological cleanup
```

### 36 Feature Groups

| Group            | Features                                    |
| ---------------- | ------------------------------------------- |
| RGB Color        | R, G, B (normalized)                        |
| HSV Color        | Hue, Saturation, Value                      |
| LAB Color        | L\*, a\*, b\* perceptual space              |
| Intensity        | Grayscale, CLAHE output                     |
| Gradient         | Sobel magnitude, Sobel X, Sobel Y           |
| Local Statistics | Mean & std at 7px, 15px, 31px windows       |
| Texture          | LBP, Blackhat at 15px, 31px, 61px scales    |
| Spatial Priors   | y_norm (normalized vertical pixel position) |
| Scene Context    | illum_norm, wet_like road indicator         |

> **Most important feature: `y_norm`** vertical pixel position, because potholes consistently appear in the lower portion of road camera images.

### Key Parameters

| Parameter                    | Value      |
| ---------------------------- | ---------- |
| Training resolution          | 320×320 px |
| App inference resolution     | 256×256 px |
| Probability threshold        | 0.80       |
| Min pothole area             | 1,200 px   |
| Max area ratio               | 0.18       |
| RDD2022 India images used    | 700 images |
| RDD2022 hard-negative pixels | 630,000    |

### Model Selection (Gradient Boosting Comparison)

| Model                 | Macro F1  | Pothole F1 | Precision | Recall |
| --------------------- | --------- | ---------- | --------- | ------ |
| **LightGBM**          | **0.827** | **0.822**  | 0.777     | 0.872  |
| Soft Ensemble (all 3) | 0.822     | 0.819      | 0.767     | 0.879  |
| CatBoost              | 0.817     | 0.815      | 0.758     | 0.883  |
| XGBoost               | 0.816     | 0.814      | 0.758     | 0.880  |

LightGBM selected as final model — best per-sample F1 and fastest training time.

### How to Run (Pipeline 2)

```bash
# Clone the repository
git clone https://github.com/ddrlve/pothole-detection
cd pothole-detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

App opens at `http://localhost:8501`. Both image and video inference modes are available.

---

## Dataset

### ARA 7.0 Road Surface (Primary Dataset)

| Split      | Images  | Purpose                                                        |
| ---------- | ------- | -------------------------------------------------------------- |
| Training   | 498     | Model training also there's 498 training mask (both pipelines) |
| Validation | 100     | Evaluation & metric computation                                |
| Test       | 295     | Final held-out test (no ground truth masks)                    |
| **Total**  | **793** |                                                                |

- Source: ARA 7.0 Road Surface competition dataset
- Contains RGB images + binary ground truth segmentation masks
- Images: Indonesian road scenes with various lighting and road conditions

### RDD2022 India (Hard-Negative Augmentation - Pipeline 2 Only)

- 700 images of road distress (cracks, patches — visually similar to potholes)
- 630,000 non-pothole pixels sampled as hard negatives
- Purpose: train the classifier to distinguish potholes from cracks and road patches
- Source: https://github.com/sekilab/RoadDamageDetector

### Class Imbalance Mitigation

- **Pipeline 1**: `class_weight='balanced'` in Random Forest — automatically weights classes inversely by frequency
- **Pipeline 2**: Curated RDD2022 hard negatives + balanced pixel sampling during training

---

## Evaluation Metrics

All models evaluated on the same 100 validation images from ARA 7.0:

| Metric             | Description                                         |
| ------------------ | --------------------------------------------------- |
| **mIoU**           | Mean Intersection over Union (background + pothole) |
| **IoU Pothole**    | IoU for the pothole class only                      |
| **Pixel Accuracy** | Fraction of correctly classified pixels             |
| **Precision**      | True positives / (true + false positives)           |
| **Recall**         | True positives / (true + false negatives)           |
| **F1 / Dice**      | Harmonic mean of precision and recall               |
| **Macro F1**       | Unweighted average F1 across both classes           |

---

## Team Contributions

| Name                                               | Contributions                                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Dian Rakhmawati Lestari (2802539085)               | Model training, model experimentation, final report, presentation, deployment, app testing |
| Fadhlan Nur Rachman (2802491690)                   | Model training, model experimentation, deployment, presentation, app testing               |
| Matthew Ken Susanto (2802407736)                   | Model experimentation, final report, presentation, app testing                             |
| Nasauramecca Nour Haqqanshah Shodiqin (2802541921) | Model experimentation, final report, presentation, demo video                              |
| Nicholas (2802424326)                              | Model experimentation, final report, presentation, app testing                             |

All members participated in weekly online meetings, peer code review, and jointly prepared the project demonstration video.

---

## References

1. Badan Pusat Statistik. (2023). _Statistik Transportasi Darat 2022_. Jakarta: BPS-Statistics Indonesia. https://www.bps.go.id
2. Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. _IEEE TPAMI, 34_(11), 2274–2282.
3. Arya, D., et al. (2022). RDD2022: A multi-national image dataset for automatic road damage detection. _arXiv:2209.08538_.
4. Breiman, L. (2001). Random forests. _Machine Learning, 45_(1), 5–32.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _KDD 2016_. https://doi.org/10.1145/2939672.2939785
6. Eriksson, J., et al. (2008). The pothole patrol. _MobiSys 2008_ (pp. 29–39).
7. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. _Annals of Statistics, 29_(5), 1189–1232.
8. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. _NeurIPS 30_. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
9. Ma, N., et al. (2022). Computer vision for road imaging and pothole detection. _Transportation Safety and Environment, 4_(4). https://doi.org/10.1093/tse/tdac026
10. Mienye, I. D., & Sun, Y. (2022). A survey of ensemble learning. _IEEE Access, 10_, 99129–99149. https://doi.org/10.1109/ACCESS.2022.3207287
11. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. _NeurIPS 31_. https://arxiv.org/abs/1706.09516
12. Safyari, Y., et al. (2024). A review of vision-based pothole detection methods. _Sensors, 24_(17), 5652. https://doi.org/10.3390/s24175652

---

_COMP7116001 — Computer Vision, School of Computer Science, BINUS University 2025/2026_
