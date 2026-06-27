# Pothole Detection — Traditional Computer Vision

**BINUS University · Semester 4 · Computer Vision Final Project**

A web application that detects **road potholes** from road photos using **traditional
computer vision — without deep learning**. The pipeline relies on illumination
normalization, road segmentation, SLIC superpixels, 9-feature extraction, and a
Random Forest classifier. Detection results from three models can be compared side by
side through an interactive Streamlit web app.

### Team Members

| No | Name | Student ID |
|----|------|------------|
| 1  | DIAN RAKHMAWATI LESTARI | 2802539085 |
| 2  | FADHLAN NUR RACHMAN | 2802491690 |
| 3  | MATTHEW KEN SUSANTO | 2802407736 |
| 4  | NASAURAMECCA NOUR HAQQANSHAH SHODIQIN | 2802541921 |
| 5  | NICHOLAS | 2802424326 |

---

## Live Demo

Live Demo Application on : https://computer-vision-pothole-project.streamlit.app/


---

## How to Run the App

**Prerequisite:** Python 3.9+ installed. A *virtual environment* is recommended so
dependencies do not clash with other projects.

### Windows (PowerShell / CMD)

```bash
# 1. Enter the project folder
cd Project_Pothole_CV

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app/app.py
```

### macOS / Linux

```bash
# 1. Enter the project folder
cd Project_Pothole_CV

# 2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python -m streamlit run app/app.py
```

The app opens automatically in the browser at `http://localhost:8501`.
To stop it: press `Ctrl + C` in the terminal. To leave the venv: `deactivate`.

## Folder Structure

```
Project_Pothole_CV/
├── app/
│   └── app.py                          # Streamlit web app (2 pages: Detection & Model Info)
├── notebooks/
│   ├── main_training.ipynb             # Training + evaluation of the 3 main models
│   └── experimentensemble_training.ipynb  # Boosting experiment (XGB/LGBM/CatBoost)
├── model/
│   ├── model_rf_9ch.joblib             # Compressed Random Forest (~51 MB) — used by the app
│   └── model_boost_9ch.joblib          # Best CatBoost (local, experiment output)
├── Dataset/
│   ├── README.md
│   ├── train/images/                   # 498 training images (train_XXX.jpg)
│   ├── train/mask/                     # 498 GT masks (mask_XXX.png)
│   └── test/images/                    # 295 test images (test_XXX.jpg)
├── result_csv/
│   ├── sp_train.csv / sp_val.csv       # Superpixel features (train/val)
│   └── *_metrics.csv                   # Per-model metrics (rf, xgboost, lightgbm, catboost)
├── docs/                               # Proposal & presentation slides (PDF)
├── .streamlit/
│   └── config.toml                     # App theme & configuration
├── requirements.txt                    # Dependencies to run the app
├── requirements-notebook.txt           # Extra dependencies for the notebooks
├── packages.txt                        # System dependencies (for Streamlit Cloud)
├── DEPLOY.md                           # Streamlit Cloud deployment guide
└── README.md
```

---

## How It Works

### Shared Pipeline (applies to all models)

```
Input image (BGR)
   │
   ├─ segment_road()        → remove sky & vegetation, keep the road area
   ├─ preprocess()          → illumination normalization + CLAHE + bilateral filter
   ├─ extract_features()    → 9 features per pixel (BGR, HSV, Gradient, Blackhat, LBP)
   ├─ compute_superpixels() → SLIC (~250 superpixels)
   ├─ aggregate_features()  → average features per superpixel
   ├─ [MODEL]               → each pixel/superpixel: pothole or not
   └─ postprocess()         → morphology + remove small areas (noise)
```

**9 features:** BGR ×3, HSV ×3, Gradient Magnitude, Blackhat, LBP.
*Blackhat* highlights small dark spots, *LBP* captures surface texture.

### Three Main Models

| Model | Type | Core Idea | Ideal Condition |
|-------|------|-----------|-----------------|
| **Adaptive Threshold** | Unsupervised | Pixels darker than their neighbours = pothole | Clean roads, dark high-contrast potholes |
| **K-Means** | Unsupervised | K-Means (K=3) on intensity; darkest cluster = pothole candidate | Roads with water puddles / dark variations |
| **Random Forest + SLIC** | Supervised | 9 features per superpixel classified by RF (200 trees) | Cracked roads & diverse conditions (most accurate) |

**Evaluation target:** mIoU ≥ 0.60 on 100 validation images (80:20 split).

### Experiment Summary (Why Random Forest?)

We tested modern boosting classifiers on the **same 9 features & pipeline**:

| Model | mIoU |
|-------|------|
| Adaptive Threshold | 0.4028 |
| K-Means | 0.4442 |
| **Random Forest (used)** | **0.5179** |
| XGBoost (threshold 0.5) | 0.4923 |
| LightGBM (threshold 0.5) | 0.4959 |
| CatBoost (calibrated threshold 0.70) | 0.5440 |

At the default threshold, boosting was actually **below** Random Forest. After
*decision threshold* calibration, CatBoost was only marginally ahead (+0.026). Since
the gain is small while Random Forest is **simpler, lighter, and easier to explain**
(feature importance), **Random Forest was chosen as the main model** of the app.

---

## Closing

This project shows that **traditional computer vision** can still detect road potholes
with reasonable results (mIoU > 0.60 in many cases) without deep learning, using a
transparent and explainable pipeline. The Streamlit web app makes it easy to compare
the three approaches directly — complete with evaluation metrics when a ground truth
mask is provided.

Thank you for trying our application <3
