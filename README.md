# Pothole Detection — Traditional Computer Vision

Final project Computer Vision — BINUS University Semester 4.

Deteksi jalan berlubang (*pothole*) menggunakan pendekatan **Computer Vision tradisional** (tanpa deep learning), dengan evaluasi menggunakan mIoU, Dice Coefficient, dan Pixel Accuracy.

---

## Struktur Folder

```
CVPothole_Project/
├── pipeline.py              # Library utama (semua fungsi CV)
├── notebooks/
│   └── main_training.ipynb  # Notebook pengembangan & evaluasi
├── app/
│   └── app.py               # Aplikasi web Streamlit
├── model/
│   ├── .gitkeep
│   └── model_rf.joblib      # Model Random Forest (di-generate saat training)
├── Dataset/
│   ├── train/images/        # 498 gambar jalan: train_XXX.jpg
│   ├── train/mask/          # 498 ground truth mask: mask_XXX.png
│   └── test/images/         # 295 gambar test: test_XXX.jpg
├── requirements.txt
└── .gitignore
```

---

## Tiga Model yang Dibandingkan

| Model | Tipe | Deskripsi |
|-------|------|-----------|
| **Baseline (Proposal)** | Unsupervised | Adaptive Thresholding pada gambar yang sudah dikoreksi iluminasi |
| **Advanced (Proposal)** | Unsupervised | K-Means clustering (k=3) — cluster paling gelap = pothole |
| **Eksperimen (Kami)** | Supervised | Random Forest pada fitur superpixel SLIC 14-channel |

### Fitur 14-Channel (Model Eksperimen)

| Ch | Fitur | Kegunaan |
|----|-------|---------|
| 0 | Intensitas | Pothole cenderung gelap |
| 1 | Local Std Dev (7x7) | Tekstur kasar |
| 2 | Local Range | Diskontinuitas kedalaman |
| 3 | Black Hat (multi-scale) | Struktur gelap kecil |
| 4 | Gabor Energy | Energi tekstur |
| 5 | LBP | Discriminator tepi vs datar |
| 6 | Gradient Magnitude (Sobel) | Tepi rim pothole |
| 7 | HSV Hue | Informasi warna |
| 8 | HSV Saturation | Saturasi warna |
| 9 | HSV Value | Kecerahan |
| 10 | Canny Edge Density | Kepadatan tepi |
| 11 | Harris Corner Response | Sudut/pojok pothole |
| 12 | LoG (Blob Detector) | Deteksi blob pothole |
| 13 | HOG Entropy | Variasi arah gradien |

---

## Setup

```bash
pip install -r requirements.txt
```

atau gunakan conda environment yang sudah ada.

---

## Cara Penggunaan

### 1. Training Model

Jalankan notebook dari awal sampai Cell 6:

```bash
jupyter notebook notebooks/main_training.ipynb
```

Cell 6 akan menyimpan model ke `model/model_rf.joblib`.

### 2. Menjalankan Aplikasi Web

```bash
streamlit run app/app.py
```

Buka browser di `http://localhost:8501`.

### 3. Evaluasi Full Dataset

Jalankan Cell 10–12 di notebook untuk evaluasi seluruh 498 gambar training dan melihat perbandingan ketiga model.

---

## Teknik Utama

**Koreksi Iluminasi (Shadow Handling)**
Membagi channel L (LAB) dengan estimasi iluminasi yang dihitung dari Gaussian blur besar. Bayangan yang jatuh ke jalan tidak lagi mempengaruhi deteksi.

**Segmentasi Jalan**
Tiga tahap fallback:
1. Eksklusikan piksel langit + vegetasi (HSV)
2. Deteksi garis horizon (Canny + Hough transform)
3. Crop bagian atas gambar

**Superpixel SLIC + Random Forest**
Gambar dibagi menjadi region homogen (superpixel). Setiap superpixel diekstrak 14-channel fiturnya, lalu diklasifikasikan oleh RF yang dilatih dari 498 ground truth mask.

---

## Target Evaluasi

Proyek ini menargetkan **mIoU >= 0.60** pada dataset training untuk model eksperimen.

---

## Dataset

Dataset terdiri dari 498 gambar jalan berlubang beserta ground truth mask biner (putih = pothole, hitam = background) dan 295 gambar test tanpa mask.
