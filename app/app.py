"""
Pothole Detection — Streamlit Web App
Pipeline: 10-channel features + simplified road segmentation + Random Forest + SLIC

Cara menjalankan (dari root project):
    streamlit run app/app.py

Syarat:
    - model/model_rf.joblib   (hasil training Cell 9 di notebook)
    - model/scaler_rf.joblib  (disimpan bersamaan dengan model)
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.segmentation import slic
from sklearn.preprocessing import StandardScaler

# ===========================================================================
# PATH DAN KONSTANTA
# ===========================================================================

ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "model" / "model_rf.joblib"
SCALER_PATH = ROOT / "model" / "scaler_rf.joblib"

# Parameter pipeline — harus sama persis dengan notebook
WORK_SCALE        = 0.5
ILLUM_BLUR_SIGMA  = 101
CLAHE_CLIP        = 2.5
CLAHE_GRID        = 8
SLIC_N_SEGMENTS   = 250
SLIC_COMPACTNESS  = 10.0
POSTPROC_MIN_AREA = 500
TOP_CROP_PCT      = 0.10
MASK_THRESHOLD    = 127

FEATURE_NAMES = [
    "Intensity", "Local Std Dev", "Local Range", "Gradient",
    "HSV Hue", "HSV Saturation", "HSV Value",
    "Harris Corners", "LoG Blob", "HOG Entropy",
]
N_FEATURES = len(FEATURE_NAMES)  # 10


# ===========================================================================
# PIPELINE — PREPROCESSING
# ===========================================================================

def normalize_illumination(gray):
    ksize        = ILLUM_BLUR_SIGMA if ILLUM_BLUR_SIGMA % 2 == 1 else ILLUM_BLUR_SIGMA + 1
    illumination = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    normalized   = gray.astype(np.float32) / (illumination + 1e-6) * 128.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def preprocess(bgr):
    L       = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L_norm  = normalize_illumination(L)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    L_clahe = clahe.apply(L_norm)
    return cv2.bilateralFilter(L_clahe, d=9, sigmaColor=50, sigmaSpace=50)


# ===========================================================================
# PIPELINE — SEGMENTASI JALAN (SEDERHANA)
# ===========================================================================

def segment_road(bgr):
    """
    Masking langit sederhana di bagian atas gambar.
    Sisanya dianggap area jalan.
    """
    h, w      = bgr.shape[:2]
    hsv       = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V   = cv2.split(hsv)
    road_mask = np.ones((h, w), dtype=np.uint8) * 255
    top_h     = h // 2
    sky       = ((H[:top_h] >= 85) & (H[:top_h] <= 140) &
                  (S[:top_h] <= 130) & (V[:top_h] >= 80))
    road_mask[:top_h][sky]        = 0
    road_mask[:int(h * TOP_CROP_PCT)] = 0
    return road_mask


# ===========================================================================
# PIPELINE — EKSTRAKSI 10 FITUR
# ===========================================================================

def extract_features(bgr, prep):
    gray = prep.astype(np.float32)

    intensity   = gray / 255.0

    g_sq        = cv2.blur(gray ** 2, (7, 7))
    g_mean      = cv2.blur(gray, (7, 7))
    local_std   = np.clip(np.sqrt(np.maximum(g_sq - g_mean ** 2, 0)) / 128.0, 0, 1)

    k9          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    local_range = (cv2.dilate(prep, k9).astype(np.float32) -
                   cv2.erode(prep, k9).astype(np.float32)) / 255.0

    sx          = cv2.Sobel(prep, cv2.CV_32F, 1, 0, ksize=3)
    sy          = cv2.Sobel(prep, cv2.CV_32F, 0, 1, ksize=3)
    grad        = np.sqrt(sx ** 2 + sy ** 2)
    gradient    = grad / grad.max() if grad.max() > 0 else grad

    hsv         = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue         = hsv[:, :, 0] / 179.0
    sat         = hsv[:, :, 1] / 255.0
    val         = hsv[:, :, 2] / 255.0

    harris      = cv2.cornerHarris(prep.astype(np.float32), 3, 3, 0.04)
    harris_pos  = np.maximum(harris, 0)
    harris_feat = harris_pos / harris_pos.max() if harris_pos.max() > 0 else harris_pos

    blurred     = cv2.GaussianBlur(prep, (9, 9), 2)
    lap         = cv2.Laplacian(blurred, cv2.CV_32F)
    lap_abs     = np.abs(lap)
    log_feat    = lap_abs / lap_abs.max() if lap_abs.max() > 0 else lap_abs

    orient      = (np.arctan2(sy, sx) + np.pi) / (2 * np.pi)
    o_mean      = cv2.blur(orient, (15, 15))
    o_sq_mean   = cv2.blur(orient ** 2, (15, 15))
    orient_var  = np.maximum(o_sq_mean - o_mean ** 2, 0)
    hog_entropy = orient_var / orient_var.max() if orient_var.max() > 0 else orient_var

    return np.stack([
        intensity, local_std, local_range, gradient,
        hue, sat, val,
        harris_feat, log_feat, hog_entropy,
    ], axis=-1).astype(np.float32)


# ===========================================================================
# PIPELINE — SUPERPIXEL DAN POST-PROCESSING
# ===========================================================================

def compute_superpixels(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return slic(
        rgb, n_segments=SLIC_N_SEGMENTS,
        compactness=SLIC_COMPACTNESS, sigma=1.0, start_label=0
    ).astype(np.int32)


def aggregate_features(feat_map, segments):
    n_seg = segments.max() + 1
    n_ch  = feat_map.shape[2]
    agg   = np.zeros((n_seg, n_ch), dtype=np.float32)
    for seg_id in range(n_seg):
        px = segments == seg_id
        if px.any():
            agg[seg_id] = feat_map[px].mean(axis=0)
    return agg


def postprocess(mask, min_area=None):
    if min_area is None:
        min_area = POSTPROC_MIN_AREA
    k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 255, cv2.FILLED)
    return result


# ===========================================================================
# PIPELINE — TIGA MODEL DETEKSI
# ===========================================================================

def detect_baseline(bgr, min_area=POSTPROC_MIN_AREA):
    road_mask = segment_road(bgr)
    prep      = preprocess(bgr)
    thresh    = cv2.adaptiveThreshold(
        prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8
    )
    pothole = cv2.bitwise_not(thresh)
    pothole = cv2.bitwise_and(pothole, road_mask)
    pothole[:int(bgr.shape[0] * TOP_CROP_PCT)] = 0
    return postprocess(pothole, min_area)


def detect_kmeans(bgr, min_area=POSTPROC_MIN_AREA):
    road_mask  = segment_road(bgr)
    h, w       = bgr.shape[:2]
    ws, hs     = int(w * WORK_SCALE), int(h * WORK_SCALE)
    bgr_s      = cv2.resize(bgr,       (ws, hs), interpolation=cv2.INTER_AREA)
    road_s     = cv2.resize(road_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)

    gray       = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY)
    clahe      = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    gray_norm  = normalize_illumination(clahe.apply(gray))

    pixels     = gray_norm.reshape(-1, 1).astype(np.float32)
    criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, K=3, bestLabels=None,
        criteria=criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )
    darkest    = int(np.argmin(centers.flatten()))
    mask_s     = (labels.reshape(hs, ws) == darkest).astype(np.uint8) * 255
    mask_s     = cv2.bitwise_and(mask_s, road_s)

    pothole    = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    pothole[:int(h * TOP_CROP_PCT)] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pothole = cv2.morphologyEx(pothole, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(pothole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(pothole)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 255, cv2.FILLED)
    return result


def detect_rf(bgr, clf, scaler, min_area=POSTPROC_MIN_AREA):
    road_mask  = segment_road(bgr)
    h, w       = bgr.shape[:2]
    ws, hs     = int(w * WORK_SCALE), int(h * WORK_SCALE)
    bgr_s      = cv2.resize(bgr,       (ws, hs), interpolation=cv2.INTER_AREA)
    road_s     = cv2.resize(road_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)

    prep         = preprocess(bgr_s)
    feat_map     = extract_features(bgr_s, prep)
    segments     = compute_superpixels(bgr_s)
    seg_feats    = aggregate_features(feat_map, segments)
    seg_scaled   = scaler.transform(seg_feats)
    pred_labels  = clf.predict(seg_scaled)

    pothole_segs = np.where(pred_labels == 1)[0]
    mask_s       = np.isin(segments, pothole_segs).astype(np.uint8) * 255
    mask_s       = cv2.bitwise_and(mask_s, road_s)

    pothole = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    pothole[:int(h * TOP_CROP_PCT)] = 0
    return postprocess(pothole, min_area)


# ===========================================================================
# METRIK EVALUASI
# ===========================================================================

def compute_metrics(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(
            gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    pred = (pred_mask > MASK_THRESHOLD).astype(np.uint8)
    gt   = (gt_mask   > MASK_THRESHOLD).astype(np.uint8)

    tp  = float(np.logical_and(pred == 1, gt == 1).sum())
    fp  = float(np.logical_and(pred == 1, gt == 0).sum())
    fn  = float(np.logical_and(pred == 0, gt == 1).sum())
    tn  = float(np.logical_and(pred == 0, gt == 0).sum())
    eps = 1e-8

    iou_pothole = tp / (tp + fp + fn + eps)
    iou_road    = tn / (tn + fp + fn + eps)
    miou        = (iou_pothole + iou_road) / 2.0
    dice        = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    acc         = (tp + tn) / (tp + tn + fp + fn + eps)
    prec        = tp / (tp + fp + eps)
    rec         = tp / (tp + fn + eps)
    f1          = (2.0 * prec * rec) / (prec + rec + eps)

    return {
        "IoU Pothole": round(iou_pothole, 4),
        "mIoU":        round(miou,        4),
        "Dice":        round(dice,        4),
        "Pixel Acc":   round(acc,         4),
        "Precision":   round(prec,        4),
        "Recall":      round(rec,         4),
        "F1":          round(f1,          4),
    }


# ===========================================================================
# HELPER — KONVERSI GAMBAR
# ===========================================================================

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_overlay(bgr, mask, color_bgr=(0, 0, 255), alpha=0.5):
    overlay = bgr.copy()
    colored = np.zeros_like(bgr)
    colored[:] = color_bgr
    px = mask > 0
    overlay[px] = (alpha * colored[px] + (1 - alpha) * bgr[px]).astype(np.uint8)
    return overlay


def read_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ===========================================================================
# LOAD MODEL (cache agar tidak reload setiap interaksi)
# ===========================================================================

@st.cache_resource
def load_rf_model():
    """Load model dan scaler dari disk. Kembalikan (clf, scaler) atau (None, None)."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    clf    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


# ===========================================================================
# STREAMLIT UI
# ===========================================================================

def main():
    st.set_page_config(
        page_title="Pothole Detection",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Header ────────────────────────────────────────────────────────────
    st.title("Pothole Detection")
    st.caption("Computer Vision Final Project")
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Pengaturan")

        method = st.radio(
            "Pilih Model",
            options=["Random Forest", "K-Means", "Adaptive Threshold"],
            index=0,
        )

        st.divider()

        st.subheader("Parameter")
        min_area = st.slider(
            "Area minimum deteksi (px)",
            min_value=100, max_value=3000, value=500, step=100,
            help="Deteksi lebih kecil dari nilai ini dianggap noise dan dihapus.",
        )

        st.divider()

        with st.expander("Tentang setiap model"):
            st.markdown("""\
**Random Forest (Eksperimen)**
Supervised — dilatih dari 498 gambar berlabel.
Menggunakan 10 fitur per superpixel SLIC.
Membutuhkan file `model/model_rf.joblib`.

**K-Means (Advanced)**
Unsupervised — tidak perlu training.
Mengelompokkan pixel menjadi 3 cluster berdasarkan kecerahan.
Cluster paling gelap = kandidat lubang.

**Adaptive Threshold (Baseline)**
Tidak perlu training.
Threshold lokal berdasarkan intensitas pixel sekitar.
Paling cepat, paling sederhana.""")

    # ── Load model jika dipilih RF ─────────────────────────────────────────
    clf, scaler = None, None
    if "Random Forest" in method:
        clf, scaler = load_rf_model()
        if clf is None:
            st.error(
                "File model tidak ditemukan.\n\n"
                f"- `{MODEL_PATH}`\n"
                f"- `{SCALER_PATH}`\n\n"
                "Latih model terlebih dahulu dengan menjalankan "
                "**Cell 9** di `notebooks/main_training.ipynb`, lalu restart app ini."
            )
            return
        if clf.n_features_in_ != N_FEATURES:
            st.error(
                f"Model memiliki **{clf.n_features_in_} fitur** "
                f"tapi pipeline sekarang menggunakan **{N_FEATURES} fitur**.\n\n"
                "Latih ulang model dengan Cell 9 di notebook (pipeline 10-fitur)."
            )
            return
        st.sidebar.success(
            f"Model: {clf.n_estimators} pohon, {clf.n_features_in_} fitur"
        )

    # ── Upload gambar ──────────────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        st.subheader("Upload Gambar Jalan")
        uploaded = st.file_uploader(
            "Pilih gambar (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    with col_info:
        st.subheader("Informasi")
        if uploaded is not None:
            bgr = read_uploaded_image(uploaded)
            if bgr is not None:
                h, w = bgr.shape[:2]
                st.metric("Lebar",   f"{w} px")
                st.metric("Tinggi",  f"{h} px")
                st.metric("Model",   method.split(" ")[0])
            else:
                st.error("Gambar tidak dapat dibaca.")
                return
        else:
            st.info("Belum ada gambar.")
            return

    # ── Tombol deteksi ─────────────────────────────────────────────────────
    if not st.button("Deteksi Lubang", type="primary", use_container_width=True):
        st.image(bgr_to_rgb(bgr), caption="Gambar yang diupload", use_container_width=True)
        return

    with st.spinner("Memproses gambar..."):
        if "Random Forest" in method:
            pred_mask = detect_rf(bgr, clf, scaler, min_area)
        elif "K-Means" in method:
            pred_mask = detect_kmeans(bgr, min_area)
        else:
            pred_mask = detect_baseline(bgr, min_area)

        road_mask = segment_road(bgr)

    # ── Tampilkan hasil ────────────────────────────────────────────────────
    st.subheader("Hasil Deteksi")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(bgr_to_rgb(bgr),
                 caption="Gambar Asli", use_container_width=True)
    with col2:
        road_overlay = make_overlay(bgr, road_mask, color_bgr=(0, 180, 0), alpha=0.4)
        st.image(bgr_to_rgb(road_overlay),
                 caption="Area Jalan", use_container_width=True)
    with col3:
        pred_overlay = make_overlay(bgr, pred_mask, color_bgr=(0, 0, 220), alpha=0.55)
        st.image(bgr_to_rgb(pred_overlay),
                 caption="Prediksi Lubang", use_container_width=True)

    # Statistik deteksi
    pothole_pct = (pred_mask > MASK_THRESHOLD).mean() * 100
    st.info(f"Area terdeteksi sebagai lubang: **{pothole_pct:.2f}%** dari total gambar")

    # Unduh mask
    ok, buf = cv2.imencode(".png", pred_mask)
    if ok:
        st.download_button(
            "Unduh Predicted Mask (PNG)",
            data=buf.tobytes(),
            file_name="predicted_mask.png",
            mime="image/png",
        )

    # ── Evaluasi dengan Ground Truth (opsional) ────────────────────────────
    st.divider()
    st.subheader("Evaluasi dengan Ground Truth (Opsional)")
    st.caption("Upload mask ground truth PNG jika tersedia untuk menghitung metrik evaluasi.")

    uploaded_gt = st.file_uploader(
        "Upload ground truth mask (PNG grayscale)",
        type=["png", "jpg"],
        key="gt_uploader",
        label_visibility="collapsed",
    )

    if uploaded_gt is not None:
        gt_bytes = np.frombuffer(uploaded_gt.read(), dtype=np.uint8)
        gt_mask  = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)

        if gt_mask is None:
            st.error("Tidak dapat membaca ground truth mask.")
        else:
            metrics = compute_metrics(pred_mask, gt_mask)

            # Metrik utama — 4 kolom baris pertama
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("IoU Pothole",  f"{metrics['IoU Pothole']:.4f}",
                      help="Metrik utama — target > 0.60")
            c2.metric("mIoU",         f"{metrics['mIoU']:.4f}",
                      help="Rata-rata IoU 2 kelas")
            c3.metric("Dice",         f"{metrics['Dice']:.4f}")
            c4.metric("Pixel Acc",    f"{metrics['Pixel Acc']:.4f}")

            # Metrik tambahan — 3 kolom baris kedua
            c5, c6, c7 = st.columns(3)
            c5.metric("Precision", f"{metrics['Precision']:.4f}")
            c6.metric("Recall",    f"{metrics['Recall']:.4f}")
            c7.metric("F1",        f"{metrics['F1']:.4f}")

            # Status target
            iou = metrics["IoU Pothole"]
            if iou >= 0.60:
                st.success(f"IoU Pothole = {iou:.4f} — target 0.60 tercapai.")
            else:
                st.warning(f"IoU Pothole = {iou:.4f} — target 0.60 belum tercapai.")

            # Visualisasi perbandingan prediksi vs GT
            st.subheader("Perbandingan Prediksi vs Ground Truth")
            ca, cb = st.columns(2)
            with ca:
                gt_overlay = make_overlay(bgr, gt_mask, color_bgr=(0, 200, 0), alpha=0.5)
                st.image(bgr_to_rgb(gt_overlay),
                         caption="Ground Truth (hijau)", use_container_width=True)
            with cb:
                st.image(bgr_to_rgb(pred_overlay),
                         caption="Prediksi (biru)", use_container_width=True)


if __name__ == "__main__":
    main()
