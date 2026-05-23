"""
Pothole Detection Pipeline
Traditional computer vision approach — no deep learning.

Three detection models for comparison:
  1. Baseline  (Proposal) : Adaptive Thresholding
  2. Advanced  (Proposal) : K-Means pixel clustering + morphological refinement
  3. Experiment (Ours)    : Random Forest on SLIC superpixel features (14 channels)
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern


# ==============================================================================
# SECTION 0: CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    # Working scale for inference (keeps computation manageable on 4K images)
    WORK_SCALE: float = 0.5

    # Working scale used during RF training (lower = faster over 498 images)
    TRAIN_SCALE: float = 0.25

    # Fraction of image height to ignore from the top (sky / horizon zone)
    TOP_CROP_PCT: float = 0.10

    # ---- Road segmentation ----
    ROAD_MIN_REMOVED_FRACTION: float = 0.05
    ROAD_SKY_HUE_LOW:  int = 85
    ROAD_SKY_HUE_HIGH: int = 135
    ROAD_SKY_SAT_MAX:  int = 130
    ROAD_SKY_VAL_MIN:  int = 90
    ROAD_VEG_HUE_LOW:  int = 20
    ROAD_VEG_HUE_HIGH: int = 95
    ROAD_VEG_SAT_MIN:  int = 30
    ROAD_MORPH_KERNEL: int = 7

    # ---- Illumination normalization (shadow correction) ----
    ILLUM_BLUR_SIGMA: int   = 101   # must be odd; large = gentle global correction
    ILLUM_EPSILON:    float = 1e-6

    # ---- CLAHE ----
    CLAHE_CLIP_LIMIT: float = 2.5
    CLAHE_TILE_GRID:  int   = 8

    # ---- Multi-scale Black Hat ----
    BH_KERNEL_SIZES: list = field(default_factory=lambda: [25, 50, 80])

    # ---- Gabor filter ----
    GABOR_KSIZE:         int   = 21
    GABOR_SIGMA:         float = 3.0
    GABOR_LAMBDA:        float = 8.0
    GABOR_GAMMA:         float = 0.5
    GABOR_N_ORIENTATIONS: int  = 6

    # ---- SLIC superpixels ----
    SLIC_N_SEGMENTS:  int   = 200
    SLIC_COMPACTNESS: float = 10.0
    SLIC_SIGMA:       float = 1.0

    # ---- K-Means (Proposal Advanced model) ----
    KMEANS_N_CLUSTERS:     int = 3
    KMEANS_N_INIT:         int = 10

    # ---- Superpixel labeling for RF training ----
    SUPERPIXEL_POTHOLE_THRESHOLD: float = 0.3

    # ---- Random Forest ----
    RF_N_ESTIMATORS:   int = 100
    RF_MAX_DEPTH:      int = 15
    RF_MIN_SAMPLES_LEAF: int = 5
    RF_RANDOM_STATE:   int = 42
    RF_N_JOBS:         int = -1
    RF_CLASS_WEIGHT:   str = "balanced"

    # ---- Post-processing ----
    POSTPROC_CLOSE_KERNEL:    int = 5
    POSTPROC_OPEN_KERNEL:     int = 5
    POSTPROC_MIN_CONTOUR_AREA: int = 500   # pixels at working scale

    # ---- Adaptive threshold (Proposal Baseline) ----
    ADAPTIVE_BLOCK_SIZE: int = 35
    ADAPTIVE_C:          int = 8

    # ---- Metric binarization — both pred and GT use the same threshold ----
    MASK_THRESHOLD: int = 127


# ==============================================================================
# SECTION 1: PREPROCESSING
# ==============================================================================

def normalize_illumination(gray: np.ndarray, cfg: Config = None) -> np.ndarray:
    """
    Koreksi bayangan dengan membagi setiap pixel terhadap estimasi
    pencahayaan lokal (hasil Gaussian blur ukuran besar).

    Area bayangan diangkat mendekati kecerahan area yang terkena cahaya,
    tanpa mengubah pola tekstur relatif di dalam setiap area.
    """
    if cfg is None:
        cfg = Config()
    sigma = cfg.ILLUM_BLUR_SIGMA
    ksize = sigma if sigma % 2 == 1 else sigma + 1
    illumination = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    normalized = gray.astype(np.float32) / (illumination + cfg.ILLUM_EPSILON) * 128.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def preprocess(bgr: np.ndarray, cfg: Config = None) -> np.ndarray:
    """
    Rangkaian preprocessing lengkap, menghasilkan satu channel uint8 (luminansi):
      1. BGR -> LAB, ambil channel L (kecerahan)
      2. Normalisasi pencahayaan (koreksi bayangan)
      3. CLAHE (peningkatan kontras lokal)
      4. Bilateral filter (mengurangi noise, mempertahankan tepi)
    """
    if cfg is None:
        cfg = Config()
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L_norm = normalize_illumination(L, cfg)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.CLAHE_CLIP_LIMIT,
        tileGridSize=(cfg.CLAHE_TILE_GRID, cfg.CLAHE_TILE_GRID)
    )
    L_clahe = clahe.apply(L_norm)
    return cv2.bilateralFilter(L_clahe, d=9, sigmaColor=50, sigmaSpace=50)


# ==============================================================================
# SECTION 2: ROAD SEGMENTATION
# ==============================================================================

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _find_horizon_line(bgr: np.ndarray) -> int:
    """Estimasi baris horizon dari tepi horizontal terkuat di bagian atas gambar."""
    h = bgr.shape[0]
    gray_upper = cv2.cvtColor(bgr[:h // 2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_upper, 50, 150)
    row_sums = np.sum(edges, axis=1)
    if row_sums.max() > 0:
        return max(int(np.argmax(row_sums)), h // 10)
    return int(h * 0.4)


def segment_road(bgr: np.ndarray, cfg: Config = None) -> tuple:
    """
    Isolasi area jalan dari langit dan vegetasi.

    Strategi (3 tahap fallback):
      1. Masking HSV langit + vegetasi → skor komponen terbesar paling bawah
      2. Fallback: estimasi horizon dari Canny
      3. Final fallback: seluruh gambar minus TOP_CROP_PCT

    Returns: (road_mask uint8, is_valid bool)
    """
    if cfg is None:
        cfg = Config()

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Masking langit (hanya separuh atas gambar)
    sky_mask = np.zeros((h, w), dtype=np.uint8)
    top_h = h // 2
    sky_top = (
        (H[:top_h] >= cfg.ROAD_SKY_HUE_LOW) & (H[:top_h] <= cfg.ROAD_SKY_HUE_HIGH) &
        (S[:top_h] <= cfg.ROAD_SKY_SAT_MAX) & (V[:top_h] >= cfg.ROAD_SKY_VAL_MIN)
    ).astype(np.uint8) * 255
    sky_mask[:top_h] = sky_top

    # Masking vegetasi
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    veg_mask = (
        (H >= cfg.ROAD_VEG_HUE_LOW) & (H <= cfg.ROAD_VEG_HUE_HIGH) &
        (S >= cfg.ROAD_VEG_SAT_MIN) & (g > r)
    ).astype(np.uint8) * 255

    road_candidate = cv2.bitwise_not(cv2.bitwise_or(sky_mask, veg_mask))

    k = cfg.ROAD_MORPH_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Pilih komponen dengan skor tertinggi: area * (centroid_y / tinggi)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        road_candidate, connectivity=8
    )
    best_label, best_score = -1, -1.0
    for lbl in range(1, num_labels):
        score = stats[lbl, cv2.CC_STAT_AREA] * (centroids[lbl][1] / h)
        if score > best_score:
            best_score, best_label = score, lbl

    if best_label == -1:
        horizon = _find_horizon_line(bgr)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[horizon:] = 255
        return road_mask, False

    road_mask = _fill_holes((labels == best_label).astype(np.uint8) * 255)
    removed = 1.0 - np.count_nonzero(road_mask) / (h * w)

    if removed < cfg.ROAD_MIN_REMOVED_FRACTION:
        horizon = _find_horizon_line(bgr)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_mask[horizon:] = 255
        return road_mask, False

    return road_mask, True


# ==============================================================================
# SECTION 3: FEATURE EXTRACTION
# ==============================================================================

def _compute_blackhat_multiscale(gray: np.ndarray, kernel_sizes: list) -> np.ndarray:
    """Max dari Black Hat transform di beberapa skala — mendeteksi struktur gelap kecil."""
    result = np.zeros_like(gray, dtype=np.float32)
    for ksize in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        result = np.maximum(result, bh.astype(np.float32))
    return result / 255.0


def _compute_gabor_energy(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """Rata-rata magnitudo respons Gabor dari N orientasi (satu channel kombinasi)."""
    angles = np.linspace(0, np.pi, cfg.GABOR_N_ORIENTATIONS, endpoint=False)
    energy = np.zeros(gray.shape, dtype=np.float32)
    for theta in angles:
        kernel = cv2.getGaborKernel(
            (cfg.GABOR_KSIZE, cfg.GABOR_KSIZE),
            cfg.GABOR_SIGMA, theta, cfg.GABOR_LAMBDA, cfg.GABOR_GAMMA, psi=0
        )
        energy += np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))
    energy /= cfg.GABOR_N_ORIENTATIONS
    m = energy.max()
    return energy / m if m > 0 else energy


def extract_features(preprocessed_gray: np.ndarray, cfg: Config = None) -> np.ndarray:
    """
    Hitung feature map 7-channel (H, W, 7), semua nilai dalam [0, 1].

    Channel:
      0  intensity    — kecerahan setelah koreksi bayangan
      1  local_std    — kekasaran tekstur (window 7x7)
      2  local_range  — kontras lokal (dilate - erode)
      3  blackhat     — skor struktur gelap (multi-skala)
      4  gabor_energy — energi tekstur rata-rata 6 orientasi
      5  lbp          — Local Binary Pattern
      6  gradient     — magnitudo tepi Sobel
    """
    if cfg is None:
        cfg = Config()

    gray = preprocessed_gray.astype(np.float32)
    intensity = gray / 255.0

    gray_sq   = cv2.blur(gray ** 2, (7, 7))
    gray_mean = cv2.blur(gray, (7, 7))
    local_std = np.clip(np.sqrt(np.maximum(gray_sq - gray_mean ** 2, 0)) / 128.0, 0, 1)

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    local_range = (cv2.dilate(preprocessed_gray, k9).astype(np.float32) -
                   cv2.erode(preprocessed_gray, k9).astype(np.float32)) / 255.0

    blackhat = _compute_blackhat_multiscale(preprocessed_gray, cfg.BH_KERNEL_SIZES)
    gabor    = _compute_gabor_energy(preprocessed_gray, cfg)

    lbp_raw = local_binary_pattern(preprocessed_gray, P=8, R=1, method="uniform").astype(np.float32)
    lbp = lbp_raw / lbp_raw.max() if lbp_raw.max() > 0 else lbp_raw

    sx = cv2.Sobel(preprocessed_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(preprocessed_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx ** 2 + sy ** 2)
    mg   = grad.max()
    gradient = grad / mg if mg > 0 else grad

    return np.stack(
        [intensity, local_std, local_range, blackhat, gabor, lbp, gradient],
        axis=-1
    ).astype(np.float32)


def extract_features_extended(
    bgr: np.ndarray,
    preprocessed_gray: np.ndarray,
    cfg: Config = None
) -> np.ndarray:
    """
    Hitung feature map 14-channel (H, W, 14), semua nilai dalam [0, 1].

    Menggabungkan 7 fitur grayscale + 3 warna HSV + 4 fitur geometri tepi/sudut:

      Ch 0-6  : Sama dengan extract_features() (intensitas, tekstur, morfologi)
      Ch 7    : HSV Hue        — distribusi warna
      Ch 8    : HSV Saturation — kejenuhan warna (lubang sering lebih kusam)
      Ch 9    : HSV Value      — kecerahan dari ruang warna
      Ch 10   : Canny Edge Density  — kepadatan tepi tajam (batas lubang)
      Ch 11   : Harris Corner Response — sudut/pertemuan tepi (rim lubang)
      Ch 12   : Laplacian of Gaussian (LoG) — detektor cekungan/blob gelap
      Ch 13   : Gradient Orientation Entropy — variasi arah gradien (HOG-like)

    Teknik-teknik yang digunakan (dari mata kuliah Computer Vision):
      - Canny (edge detection)
      - Harris corner detector
      - Laplacian of Gaussian = blob detector
      - Histogram of Oriented Gradients (HOG) — diperkirakan via variance orientasi
    """
    if cfg is None:
        cfg = Config()

    # Ch 0-6: fitur grayscale dasar
    gray_features = extract_features(preprocessed_gray, cfg)

    # Ch 7-9: warna dari HSV
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch = hsv[:, :, 0] / 179.0   # OpenCV: Hue range 0-179
    s_ch = hsv[:, :, 1] / 255.0
    v_ch = hsv[:, :, 2] / 255.0

    # Ch 10: Canny Edge Density
    # Tepi tajam menandai batas antara lubang dan aspal normal.
    # Canny menggunakan Gaussian smoothing + gradient magnitude + non-max suppression.
    canny = cv2.Canny(preprocessed_gray, threshold1=50, threshold2=150)
    canny_feat = canny.astype(np.float32) / 255.0

    # Ch 11: Harris Corner Response
    # Sudut (corner) terbentuk di titik pertemuan dua atau lebih tepi.
    # Rim lubang sering menghasilkan respons corner yang tinggi.
    gray_f32 = preprocessed_gray.astype(np.float32)
    harris   = cv2.cornerHarris(gray_f32, blockSize=3, ksize=3, k=0.04)
    harris_pos = np.maximum(harris, 0)
    hm = harris_pos.max()
    harris_feat = harris_pos / hm if hm > 0 else harris_pos

    # Ch 12: Laplacian of Gaussian (LoG) — Blob Detector
    # LoG mendeteksi cekungan (negative curvature) dan tonjolan (positive curvature).
    # Lubang jalan adalah cekungan gelap → memberikan respons LoG yang tinggi.
    blurred_log = cv2.GaussianBlur(preprocessed_gray, (9, 9), 2)
    laplacian   = cv2.Laplacian(blurred_log, cv2.CV_32F)
    lap_mag = np.abs(laplacian)
    lm = lap_mag.max()
    log_feat = lap_mag / lm if lm > 0 else lap_mag

    # Ch 13: Gradient Orientation Entropy (HOG-like)
    # Lubang punya tepi dari berbagai arah sekaligus → variansi orientasi tinggi.
    # Permukaan datar mulus punya gradien seragam → variansi orientasi rendah.
    sx_full = cv2.Sobel(preprocessed_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy_full = cv2.Sobel(preprocessed_gray, cv2.CV_32F, 0, 1, ksize=3)
    orientation = (np.arctan2(sy_full, sx_full) + np.pi) / (2 * np.pi)  # [0, 1]
    orient_mean    = cv2.blur(orientation, (15, 15))
    orient_sq_mean = cv2.blur(orientation ** 2, (15, 15))
    orient_var     = np.maximum(orient_sq_mean - orient_mean ** 2, 0)
    ov_max = orient_var.max()
    hog_entropy = orient_var / ov_max if ov_max > 0 else orient_var

    return np.concatenate([
        gray_features,
        h_ch[..., None], s_ch[..., None], v_ch[..., None],
        canny_feat[..., None],
        harris_feat[..., None],
        log_feat[..., None],
        hog_entropy[..., None],
    ], axis=-1).astype(np.float32)


# Nama tiap channel — dipakai di notebook untuk visualisasi dan feature importance
FEATURE_NAMES = [
    "Intensity", "Local Std Dev", "Local Range", "Black Hat",
    "Gabor Energy", "LBP", "Gradient Magnitude",
    "HSV Hue", "HSV Saturation", "HSV Value",
    "Canny Edges", "Harris Corners", "LoG (Blob)", "HOG Entropy",
]


# ==============================================================================
# SECTION 4: SUPERPIXEL AGGREGATION
# ==============================================================================

def compute_superpixels(bgr_small: np.ndarray, cfg: Config = None) -> np.ndarray:
    """Hitung label superpixel SLIC. Mengembalikan label map integer (H, W)."""
    if cfg is None:
        cfg = Config()
    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
    return slic(
        rgb,
        n_segments=cfg.SLIC_N_SEGMENTS,
        compactness=cfg.SLIC_COMPACTNESS,
        sigma=cfg.SLIC_SIGMA,
        start_label=0
    ).astype(np.int32)


def aggregate_superpixel_features(
    feature_map: np.ndarray,
    segments: np.ndarray
) -> np.ndarray:
    """
    Rata-ratakan tiap channel fitur di dalam setiap superpixel.
    Mengembalikan array (n_superpixels, n_channels).
    """
    n_seg = segments.max() + 1
    n_ch  = feature_map.shape[2]
    agg   = np.zeros((n_seg, n_ch), dtype=np.float32)
    for seg_id in range(n_seg):
        px = segments == seg_id
        if px.any():
            agg[seg_id] = feature_map[px].mean(axis=0)
    return agg


def label_superpixels(
    segments: np.ndarray,
    gt_mask_small: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Beri label biner (0=jalan, 1=lubang) ke setiap superpixel.
    Superpixel dianggap lubang jika >= threshold pixel-nya ada di GT mask.
    """
    gt_binary = (gt_mask_small > 127).astype(np.uint8)
    n_seg  = segments.max() + 1
    labels = np.zeros(n_seg, dtype=np.int32)
    for seg_id in range(n_seg):
        px = segments == seg_id
        if px.any() and gt_binary[px].mean() >= threshold:
            labels[seg_id] = 1
    return labels


# ==============================================================================
# SECTION 5: MODEL 1 — PROPOSAL BASELINE (ADAPTIVE THRESHOLDING)
# ==============================================================================

def detect_baseline(
    bgr: np.ndarray,
    road_mask: np.ndarray,
    cfg: Config = None
) -> np.ndarray:
    """
    Model Baseline (sesuai proposal): Adaptive Thresholding.

    Langkah:
      1. Preprocessing (normalisasi pencahayaan + CLAHE + bilateral)
      2. Adaptive Gaussian threshold → temukan area gelap lokal
      3. Invert (area gelap di bawah threshold = lubang)
      4. Terapkan road mask + ROI crop
      5. Post-processing morfologi + filter kontur kecil

    Kelebihan  : cepat, deterministik, tidak perlu training
    Kekurangan : hanya memakai satu sinyal (kecerahan lokal),
                 bayangan dan noda mudah salah terdeteksi sebagai lubang
    """
    if cfg is None:
        cfg = Config()

    preprocessed = preprocess(bgr, cfg)
    thresh = cv2.adaptiveThreshold(
        preprocessed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        cfg.ADAPTIVE_BLOCK_SIZE, cfg.ADAPTIVE_C
    )
    pothole_mask = cv2.bitwise_not(thresh)
    pothole_mask = cv2.bitwise_and(pothole_mask, road_mask)
    pothole_mask = apply_roi_crop(pothole_mask, cfg.TOP_CROP_PCT)
    return postprocess_mask(pothole_mask, cfg)


def run_baseline(bgr: np.ndarray, cfg: Config = None) -> tuple:
    """Pipeline end-to-end Baseline. Returns (pred_mask, road_mask, road_valid)."""
    if cfg is None:
        cfg = Config()
    road_mask, road_valid = segment_road(bgr, cfg)
    return detect_baseline(bgr, road_mask, cfg), road_mask, road_valid


# ==============================================================================
# SECTION 6: MODEL 2 — PROPOSAL ADVANCED (K-MEANS CLUSTERING)
# ==============================================================================

def detect_kmeans(
    bgr: np.ndarray,
    road_mask: np.ndarray,
    cfg: Config = None
) -> np.ndarray:
    """
    Model Advanced sesuai proposal: K-Means pixel clustering.

    Implementasi tepat sesuai deskripsi di proposal:
      Preprocessing  : Grayscale + CLAHE + normalisasi pencahayaan
      Clustering     : K-Means (K=3) pada nilai intensitas pixel
      Seleksi cluster: Ambil cluster dengan centroid paling gelap (= lubang)
      Post-processing: Morphological CLOSE + contour filtering

    Kelebihan  : lebih fleksibel dari threshold tunggal
    Kekurangan : unsupervised (tidak belajar dari GT mask),
                 cluster "gelap" bisa saja bayangan atau aspal retak,
                 bukan hanya lubang
    """
    if cfg is None:
        cfg = Config()

    h_orig, w_orig = bgr.shape[:2]
    w_s = int(w_orig * cfg.WORK_SCALE)
    h_s = int(h_orig * cfg.WORK_SCALE)

    bgr_s  = cv2.resize(bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)
    road_s = cv2.resize(road_mask, (w_s, h_s), interpolation=cv2.INTER_NEAREST)

    # Preprocessing sesuai proposal: Grayscale + CLAHE
    gray = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.CLAHE_CLIP_LIMIT,
        tileGridSize=(cfg.CLAHE_TILE_GRID, cfg.CLAHE_TILE_GRID)
    )
    gray_eq   = clahe.apply(gray)
    gray_norm = normalize_illumination(gray_eq, cfg)

    # K-Means pada nilai intensitas pixel
    pixels   = gray_norm.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, K=cfg.KMEANS_N_CLUSTERS,
        bestLabels=None, criteria=criteria,
        attempts=cfg.KMEANS_N_INIT,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # Cluster paling gelap = kandidat lubang
    darkest_cluster = int(np.argmin(centers.flatten()))
    label_map       = labels.reshape(h_s, w_s)
    mask_s = (label_map == darkest_cluster).astype(np.uint8) * 255

    # Terapkan road mask
    mask_s = cv2.bitwise_and(mask_s, road_s)

    # Upscale ke resolusi asli
    pothole_mask = cv2.resize(mask_s, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    pothole_mask = apply_roi_crop(pothole_mask, cfg.TOP_CROP_PCT)

    # Morphological CLOSE (sesuai proposal) + filter kontur
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.POSTPROC_CLOSE_KERNEL,) * 2
    )
    pothole_mask = cv2.morphologyEx(pothole_mask, cv2.MORPH_CLOSE, close_k)
    contours, _  = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(pothole_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= cfg.POSTPROC_MIN_CONTOUR_AREA:
            cv2.drawContours(result, [cnt], -1, 255, cv2.FILLED)
    return result


def run_kmeans(bgr: np.ndarray, cfg: Config = None) -> tuple:
    """Pipeline end-to-end K-Means (Proposal Advanced). Returns (pred_mask, road_mask, road_valid)."""
    if cfg is None:
        cfg = Config()
    road_mask, road_valid = segment_road(bgr, cfg)
    return detect_kmeans(bgr, road_mask, cfg), road_mask, road_valid


# ==============================================================================
# SECTION 7: MODEL 3 — EKSPERIMEN (RANDOM FOREST + SLIC + 14 FITUR)
# ==============================================================================

def train_model(
    train_img_dir: Path,
    train_mask_dir: Path,
    cfg: Config = None,
    max_images: int = None,
    verbose: bool = True
) -> RandomForestClassifier:
    """
    Latih Random Forest pada fitur superpixel dari gambar training berlabel.

    Perbedaan kunci dari K-Means:
      - K-Means tidak pernah melihat GT mask → tidak tahu seperti apa lubang
      - RF dilatih dari 498 GT mask nyata → belajar kombinasi fitur yang
        membedakan lubang dari bayangan, retak, dan aspal normal

    Fitur: 14 channel (7 grayscale + 3 HSV warna + 4 tepi/sudut/bentuk)
    """
    if cfg is None:
        cfg = Config()

    all_imgs = sorted(train_img_dir.glob("*.jpg"))
    if max_images is not None:
        all_imgs = all_imgs[:max_images]

    all_features, all_labels = [], []

    for i, img_path in enumerate(all_imgs):
        idx_str   = img_path.stem.split("_")[1]
        mask_path = train_mask_dir / f"mask_{idx_str}.png"

        bgr  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            continue

        scale = cfg.TRAIN_SCALE
        h_s = int(bgr.shape[0] * scale)
        w_s = int(bgr.shape[1] * scale)
        bgr_s  = cv2.resize(bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, (w_s, h_s), interpolation=cv2.INTER_NEAREST)

        prep     = preprocess(bgr_s, cfg)
        feat_map = extract_features_extended(bgr_s, prep, cfg)
        segments = compute_superpixels(bgr_s, cfg)
        seg_feats  = aggregate_superpixel_features(feat_map, segments)
        seg_labels = label_superpixels(segments, mask_s, cfg.SUPERPIXEL_POTHOLE_THRESHOLD)

        all_features.append(seg_feats)
        all_labels.append(seg_labels)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Ekstraksi fitur: {i + 1}/{len(all_imgs)} gambar selesai")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    if verbose:
        n_pothole = y.sum()
        print(f"\nData training: {len(X)} superpixel")
        print(f"  Lubang   : {n_pothole} ({n_pothole/len(y)*100:.1f}%)")
        print(f"  Jalan    : {len(y)-n_pothole} ({(len(y)-n_pothole)/len(y)*100:.1f}%)")
        print("Melatih Random Forest...")

    clf = RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        class_weight=cfg.RF_CLASS_WEIGHT,
        random_state=cfg.RF_RANDOM_STATE,
        n_jobs=cfg.RF_N_JOBS,
    )
    clf.fit(X, y)

    if verbose:
        print("Training selesai.")
    return clf


def save_model(clf: RandomForestClassifier, path: str = "model/model_rf.joblib") -> None:
    """Simpan model Random Forest ke disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load_model(path: str = "model/model_rf.joblib") -> RandomForestClassifier:
    """Load model Random Forest dari disk."""
    return joblib.load(path)


def detect_rf(
    bgr: np.ndarray,
    road_mask: np.ndarray,
    clf: RandomForestClassifier,
    cfg: Config = None
) -> np.ndarray:
    """
    Model Eksperimen: Random Forest pada fitur superpixel SLIC (14 channel).

    Langkah:
      1. Downscale sebesar WORK_SCALE
      2. Preprocessing (koreksi bayangan + CLAHE + bilateral)
      3. Ekstraksi 14-channel feature map
      4. Hitung superpixel SLIC
      5. Rata-rata fitur per superpixel
      6. clf.predict() → label lubang/jalan per superpixel
      7. Bangun mask pixel dari superpixel berlabel lubang
      8. Upscale + road mask + ROI crop + post-processing
    """
    if cfg is None:
        cfg = Config()

    h_orig, w_orig = bgr.shape[:2]
    w_s = int(w_orig * cfg.WORK_SCALE)
    h_s = int(h_orig * cfg.WORK_SCALE)

    bgr_s  = cv2.resize(bgr, (w_s, h_s), interpolation=cv2.INTER_AREA)
    road_s = cv2.resize(road_mask, (w_s, h_s), interpolation=cv2.INTER_NEAREST)

    prep      = preprocess(bgr_s, cfg)
    feat_map  = extract_features_extended(bgr_s, prep, cfg)
    segments  = compute_superpixels(bgr_s, cfg)
    seg_feats = aggregate_superpixel_features(feat_map, segments)

    pred_labels  = clf.predict(seg_feats)
    pothole_segs = np.where(pred_labels == 1)[0]
    mask_small   = np.isin(segments, pothole_segs).astype(np.uint8) * 255

    mask_small   = cv2.bitwise_and(mask_small, road_s)
    pothole_mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    pothole_mask = apply_roi_crop(pothole_mask, cfg.TOP_CROP_PCT)
    return postprocess_mask(pothole_mask, cfg)


def run_rf(bgr: np.ndarray, clf: RandomForestClassifier, cfg: Config = None) -> tuple:
    """Pipeline end-to-end RF (Eksperimen). Returns (pred_mask, road_mask, road_valid)."""
    if cfg is None:
        cfg = Config()
    road_mask, road_valid = segment_road(bgr, cfg)
    return detect_rf(bgr, road_mask, clf, cfg), road_mask, road_valid


# ==============================================================================
# SECTION 8: POST-PROCESSING
# ==============================================================================

def apply_roi_crop(mask: np.ndarray, top_crop_pct: float) -> np.ndarray:
    """Hapus bagian atas mask (zona langit/horizon) untuk mengurangi false positive."""
    result = mask.copy()
    result[:int(mask.shape[0] * top_crop_pct)] = 0
    return result


def postprocess_mask(mask: np.ndarray, cfg: Config = None) -> np.ndarray:
    """
    Perbaikan morfologi + hapus kontur kecil:
      1. MORPH_CLOSE — mengisi celah kecil di dalam area lubang
      2. MORPH_OPEN  — menghapus noise berbentuk filamen tipis
      3. Filter kontur — hapus deteksi yang terlalu kecil
    """
    if cfg is None:
        cfg = Config()

    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.POSTPROC_CLOSE_KERNEL,) * 2)
    ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.POSTPROC_OPEN_KERNEL,) * 2)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, ok)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) >= cfg.POSTPROC_MIN_CONTOUR_AREA:
            cv2.drawContours(result, [cnt], -1, 255, cv2.FILLED)
    return result


# ==============================================================================
# SECTION 9: METRIK EVALUASI
# ==============================================================================

def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: int = None
) -> dict:
    """
    Hitung metrik segmentasi. Kedua mask di-binarisasi dengan threshold yang sama.

    mIoU = rata-rata IoU kelas lubang (foreground) DAN kelas jalan (background),
           sesuai evaluasi segmentasi semantik 2-kelas standar.

    Returns dict: mIoU, Dice, PixelAccuracy, Precision, Recall, F1
    """
    if threshold is None:
        threshold = Config().MASK_THRESHOLD

    pred = (pred_mask > threshold).astype(np.uint8)
    gt   = (gt_mask   > threshold).astype(np.uint8)

    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    tn = float(np.logical_and(pred == 0, gt == 0).sum())

    eps    = 1e-8
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fn + fp + eps)
    miou   = (iou_fg + iou_bg) / 2.0
    dice   = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    acc    = (tp + tn) / (tp + tn + fp + fn + eps)
    prec   = tp / (tp + fp + eps)
    rec    = tp / (tp + fn + eps)
    f1     = (2.0 * prec * rec) / (prec + rec + eps)

    return {
        "mIoU":          round(miou, 4),
        "Dice":          round(dice, 4),
        "PixelAccuracy": round(acc,  4),
        "Precision":     round(prec, 4),
        "Recall":        round(rec,  4),
        "F1":            round(f1,   4),
    }


# ==============================================================================
# SECTION 10: VISUALISASI
# ==============================================================================

def make_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple = (0, 0, 255),
    alpha: float = 0.5
) -> np.ndarray:
    """Tumpangkan mask berwarna di atas gambar BGR dengan transparansi."""
    overlay = bgr.copy()
    colored = np.zeros_like(bgr)
    colored[:] = color_bgr
    px = mask > 0
    overlay[px] = (alpha * colored[px] + (1 - alpha) * bgr[px]).astype(np.uint8)
    return overlay


def plot_pipeline_result(
    bgr: np.ndarray,
    road_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray = None,
    metrics: dict = None,
    title: str = ""
) -> plt.Figure:
    """
    Visualisasi hasil pipeline.
    Panel: Original | Road Mask | Prediksi [| Ground Truth]
    """
    n = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[1].imshow(cv2.cvtColor(make_overlay(bgr, road_mask, (0, 255, 0), 0.4), cv2.COLOR_BGR2RGB))
    axes[1].set_title("Road Mask (hijau)")
    axes[2].imshow(cv2.cvtColor(make_overlay(bgr, pred_mask, (0, 0, 255), 0.5), cv2.COLOR_BGR2RGB))
    axes[2].set_title("Prediksi Lubang (merah)")

    if gt_mask is not None:
        axes[3].imshow(cv2.cvtColor(make_overlay(bgr, gt_mask, (0, 255, 255), 0.5), cv2.COLOR_BGR2RGB))
        axes[3].set_title("Ground Truth (cyan)")

    for ax in axes:
        ax.axis("off")

    if metrics:
        s = "  |  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        fig.suptitle(f"{title}\n{s}" if title else s, fontsize=9)
    elif title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_comparison(
    bgr: np.ndarray,
    results: dict,
    gt_mask: np.ndarray = None
) -> plt.Figure:
    """
    Tampilkan perbandingan beberapa model dalam satu baris.
    results = {"Nama Model": pred_mask, ...}
    """
    n = len(results) + (2 if gt_mask is not None else 1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(cv2.cvtColor(make_overlay(bgr, gt_mask, (0, 255, 255), 0.5), cv2.COLOR_BGR2RGB))
        axes[1].set_title("Ground Truth (cyan)")
        axes[1].axis("off")
        offset = 2
    else:
        offset = 1

    for i, (name, pred_mask) in enumerate(results.items()):
        m = compute_metrics(pred_mask, gt_mask) if gt_mask is not None else {}
        title = f"{name}"
        if m:
            title += f"\nmIoU={m['mIoU']:.3f} | Dice={m['Dice']:.3f}"
        axes[offset + i].imshow(
            cv2.cvtColor(make_overlay(bgr, pred_mask, (0, 0, 255), 0.5), cv2.COLOR_BGR2RGB)
        )
        axes[offset + i].set_title(title, fontsize=9)
        axes[offset + i].axis("off")

    plt.tight_layout()
    return fig


def plot_evaluation_summary(df: pd.DataFrame, title: str = "") -> plt.Figure:
    """Grafik ringkasan evaluasi: bar mean metrik + distribusi mIoU per gambar."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    means = df[["mIoU", "Dice", "PixelAccuracy"]].mean()
    bars  = ax1.bar(means.index, means.values, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_title(f"Rata-Rata Metrik Evaluasi{' — ' + title if title else ''}")
    ax1.axhline(0.6, color="red", linestyle="--", linewidth=1, label="Target mIoU=0.6")
    ax1.legend()
    for bar, val in zip(bars, means.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9)

    sorted_iou = df["mIoU"].sort_values().reset_index(drop=True)
    ax2.bar(range(len(sorted_iou)), sorted_iou.values, color="#2196F3", alpha=0.7)
    ax2.axhline(sorted_iou.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {sorted_iou.mean():.3f}")
    ax2.axhline(0.6, color="orange", linestyle=":", linewidth=1.5, label="Target = 0.60")
    ax2.set_xlabel("Indeks gambar (diurutkan berdasarkan mIoU)")
    ax2.set_ylabel("mIoU")
    ax2.set_title("Distribusi mIoU Per Gambar")
    ax2.legend()

    plt.tight_layout()
    return fig
