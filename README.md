# Pothole Detection — Traditional Computer Vision

BINUS University · Semester 4 · Computer Vision Final Project

Deteksi jalan berlubang menggunakan **computer vision tradisional** (tanpa deep learning) — normalisasi iluminasi, superpixel SLIC, ekstraksi 9 fitur, dan Random Forest.

---

## Struktur Folder

```
Project_Pothole_CV/
├── app/
│   └── app.py                    # Streamlit web app (self-contained)
├── notebooks/
│   └── main_training.ipynb       # Training & evaluasi semua model
├── model/
│   └── model_rf_9ch.joblib      # Hasil training (di-generate, tidak di-commit)
├── Dataset/
│   ├── train/images/             # 498 gambar: train_XXX.jpg
│   ├── train/mask/               # 498 GT mask: mask_XXX.png
│   └── test/images/              # 295 gambar test: test_XXX.jpg
├── results/                      # CSV fitur & metrik (di-generate)
├── requirements.txt
└── CLAUDE.md                     # Panduan untuk Claude Code
```

---

## Tiga Model

| Model | Tipe | Cara Kerja |
|-------|------|------------|
| **Baseline** | Unsupervised | Adaptive Thresholding pada citra yang sudah dikoreksi iluminasi |
| **Advanced** | Unsupervised | K-Means (K=3) — cluster paling gelap = kandidat lubang |
| **Eksperimen** | Supervised | Random Forest pada 9 fitur superpixel SLIC |

**9 Fitur (RF):** BGR ×3, HSV ×3, Gradient Magnitude, Blackhat, LBP

**Target evaluasi:** mIoU ≥ 0.60 pada 100 gambar val (split 80:20)

---

## Setup

**Prasyarat:** Python 3.9+

```bash
# 1. Clone / buka folder project
cd Project_Pothole_CV

# 2. Install dependensi
pip install -r requirements.txt
```

> Alternatif conda: `conda activate ai_core` (jika environment sudah ada)

---

## Cara Penggunaan

### 1. Training Model

Jalankan notebook dari atas ke bawah sampai selesai Bagian 5:

```bash
jupyter notebook notebooks/main_training.ipynb
```

Urutan penting:
- **Bagian 2** — build CSV fitur superpixel (~5–8 menit, sekali jalan)
- **Bagian 5** — training RF dan simpan model ke `model/model_rf_9ch.joblib`

### 2. Menjalankan Aplikasi Web (Streamlit)

Pastikan model sudah di-training terlebih dahulu, lalu:

```bash
# Semua OS (Windows / macOS / Linux)
python -m streamlit run app/app.py
```

Buka browser di: `http://localhost:8501`

> **Catatan per OS:**
> - **Windows** — gunakan `python -m streamlit run app/app.py` (bukan `streamlit run` langsung, agar PATH tidak perlu dikonfigurasi)
> - **macOS / Linux** — bisa juga `streamlit run app/app.py` jika streamlit sudah ada di PATH

**Fitur aplikasi:**
- Upload gambar jalan (JPG/PNG)
- Pilih model: Random Forest, K-Means, atau Adaptive Threshold
- Atur area minimum deteksi (slider)
- Lihat overlay hasil deteksi + area jalan
- Upload ground truth mask untuk evaluasi metrik (opsional)
- Download predicted mask (PNG)

### 3. Evaluasi Full Dataset

Jalankan seluruh notebook untuk melihat perbandingan ketiga model pada 100 gambar val, termasuk tabel metrik dan visualisasi.

---

## Pipeline Singkat

```
Input BGR
   │
   ├─ segment_road()     → road mask (hapus langit + vegetasi)
   ├─ preprocess()       → normalisasi iluminasi + CLAHE + bilateral filter
   ├─ extract_features() → 9-channel feature map per piksel
   ├─ compute_superpixels() → SLIC label map (~250 superpixel)
   ├─ aggregate_features()  → rata-rata fitur per superpixel
   ├─ RF.predict()       → label tiap superpixel (0=jalan, 1=lubang)
   └─ postprocess()      → morphological close/open + filter area kecil
```

---

## Requirements

```
opencv-python >= 4.8.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
scikit-image >= 0.21.0
matplotlib >= 3.7.0
pandas >= 2.0.0
streamlit >= 1.28.0
Pillow >= 10.0.0
joblib >= 1.3.0
```
