# Pothole Segmentation — Kaggle Experiment (Dian)

Notebook ini dijalankan di **Kaggle** sebagai eksperimen lanjutan dari proyek utama, menggunakan dataset kompetisi **Data Science ARA 7.0** dan data eksternal **RDD2022 India** sebagai negative-only augmentation.

Eksperimen ini dilakukan atas rekomendasi dosen untuk mencoba model-model boosting (LightGBM, XGBoost, CatBoost) sebagai alternatif dari Random Forest yang dipakai di app utama.

---

## Dataset

| Dataset                  | Sumber                                                                             | Kegunaan                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Data Science ARA 7.0** | Kaggle Competition (`/kaggle/input/competitions/data-science-ara-7-0`)             | Data utama: 498 train images + mask, 295 test images            |
| **RDD2022 India**        | Kaggle Dataset (`https://www.kaggle.com/datasets/fadhlannurrachman/rdd2022-india`) | Negative-only augmentation (max 700 gambar, tanpa mask pothole) |

RDD2022 India tidak dipakai sebagai data training berlabel, melainkan hanya sebagai **negative sample tambahan** (aspal mulus, bayangan, jalan basah) untuk membantu model mengurangi false positive. Karena dataset ini cukup sulit dan tidak memiliki mask segmentasi yang sesuai, maka hanya diambil piksel negatifnya saja berdasarkan bounding box annotation VOC-format.

---

## Pipeline

```
Input image (BGR)
   │
   ├─ conservative_road_mask()   → masking langit & vegetasi
   ├─ build_feature_map()        → 20+ fitur per piksel
   ├─ sample_pixels()            → balanced sampling (50% pos, 50% neg dengan hard-negative mining)
   ├─ [MODEL]                    → LightGBM / XGBoost / CatBoost / Soft Ensemble
   ├─ postprocess_probability()  → threshold tuning + morphology + remove small areas
   └─ RLE encoding               → submission.csv untuk Kaggle
```

### Fitur (20+)

Dibandingkan pipeline utama (9 fitur), notebook ini menggunakan fitur yang lebih kaya:

| Grup        | Fitur                                                             |
| ----------- | ----------------------------------------------------------------- |
| Warna       | RGB ×3, HSV ×3, LAB ×3                                            |
| Tekstur     | Blackhat (kernel 15, 31, 61), LBP                                 |
| Pencahayaan | CLAHE, illumination-normalized gray, local mean & std (7px, 15px) |
| Gabor       | 4 orientasi (0°, 45°, 90°, 135°)                                  |
| Konteks     | Posisi (x_norm, y_norm, center_x), wet_like, shadow_like          |
| Segmentasi  | road_mask                                                         |

Fitur terpenting (LGBM feature importance): `y_norm`, `illum_norm`, `blackhat61`, `wet_like`, `x_norm`.

---

## Model

Semua model dilatih pada fitur yang **sama** (classical ML, tanpa deep learning).

### Hasil Sample-Level (pixel sampling, sebelum post-processing)

| Model         | Macro F1   | Pothole F1 | Precision  | Recall     |
| ------------- | ---------- | ---------- | ---------- | ---------- |
| **LightGBM**  | **0.8266** | **0.8217** | **0.7770** | **0.8718** |
| Soft Ensemble | 0.8223     | 0.8193     | 0.7670     | 0.8792     |
| CatBoost      | 0.8170     | 0.8154     | 0.7577     | 0.8825     |
| XGBoost       | 0.8162     | 0.8143     | 0.7578     | 0.8800     |

LightGBM dipilih sebagai model utama (fast model) karena mencapai macro F1 tertinggi.

### Konfigurasi Post-Processing Terbaik (tuned via grid search)

| Parameter      | Nilai   |
| -------------- | ------- |
| Threshold      | 0.80    |
| Min area       | 1200 px |
| Close kernel   | 3       |
| Open kernel    | 3       |
| Fill holes     | True    |
| Max area ratio | 0.18    |

### Hasil Validasi Akhir (100 gambar, 80:20 split)

| Metrik         | Nilai      |
| -------------- | ---------- |
| **mIoU**       | **0.5918** |
| IoU pothole    | 0.3201     |
| IoU background | 0.8635     |
| Dice           | 0.4506     |
| Pixel Accuracy | 0.8741     |
| Precision      | 0.4522     |
| Recall         | 0.6187     |
| Macro F1       | 0.6871     |

### Overfit Check (30 train samples vs validation)

| Split        | mIoU   | IoU pothole | Macro F1 |
| ------------ | ------ | ----------- | -------- |
| Train subset | 0.6177 | 0.3656      | 0.7185   |
| Validation   | 0.5918 | 0.3201      | 0.6871   |

Gap terkontrol → model tidak overfit.

---

## Output Artifacts

Notebook menyimpan output ke `/kaggle/working/pothole_output/`:

| File                          | Keterangan                                      |
| ----------------------------- | ----------------------------------------------- |
| `pothole_model_fast.pkl`      | LightGBM (deployment utama)                     |
| `pothole_model_accuracy.pkl`  | LightGBM (accuracy mode)                        |
| `pothole_model.pkl`           | Model fallback                                  |
| `pothole_config.json`         | Konfigurasi post-processing terbaik             |
| `submission.csv`              | Prediksi test set dalam format RLE untuk Kaggle |
| `validation_metrics.csv`      | Metrik validasi akhir                           |
| `train_val_metrics.csv`       | Perbandingan train vs val (overfit check)       |
| `feature_importance_lgbm.csv` | Feature importance LightGBM                     |
| `threshold_search.csv`        | Hasil grid search post-processing               |

Setelah notebook selesai, download `pothole_artifacts.zip` dan jalankan Streamlit app lokal:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Catatan

- RDD2022 India diuji tapi **tidak memberikan improvement signifikan** karena karakteristik jalannya berbeda dengan dataset ARA. Akhirnya tetap dipakai tapi dibatasi (max 700 gambar, hanya piksel negatif di luar bounding box pothole).
- Soft Ensemble (rata-rata probabilitas LGBM + XGB + CatBoost) tidak melampaui LightGBM tunggal, sehingga LightGBM saja yang dipakai sebagai deployment model.
- Nilai mIoU 0.5918 ini menggunakan pipeline dan dataset yang **sama** dengan app utama, namun dengan fitur lebih banyak dan post-processing yang lebih teliti.
