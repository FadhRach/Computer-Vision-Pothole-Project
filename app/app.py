"""
Pothole Detection — Streamlit Web App
Self-contained: semua fungsi pipeline ada di file ini.

Jalankan dari root project:
    streamlit run app/app.py

Syarat:
    model/model_rf_9ch.joblib  (hasil training di notebook Bagian 5)
"""

import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern


# ===========================================================================
# KONFIGURASI
# ===========================================================================

ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'model' / 'model_rf_9ch.joblib'

WORK_SCALE   = 0.5
ILLUM_SIGMA  = 101
CLAHE_CLIP   = 2.5
CLAHE_GRID   = 8
TOP_CROP_PCT = 0.10
MASK_THRESH  = 127
SLIC_N_SEG   = 250
SLIC_COMPACT = 10.0
MIN_AREA     = 500
POSTPROC_K   = 5

# Segmentasi jalan
SKY_HUE_LOW  = 85
SKY_HUE_HIGH = 140
SKY_SAT_MAX  = 130
SKY_VAL_MIN  = 80
VEG_HUE_LOW  = 20
VEG_HUE_HIGH = 85
VEG_SAT_MIN  = 40
ROAD_MORPH_K = 7

FEATURE_NAMES = [
    'BGR Blue', 'BGR Green', 'BGR Red',
    'HSV Hue', 'HSV Sat', 'HSV Val',
    'Gradient Mag', 'Blackhat', 'LBP',
]
N_FEATURES = 9


# ===========================================================================
# PREPROCESSING
# ===========================================================================

def normalize_illumination(gray):
    ksize = ILLUM_SIGMA if ILLUM_SIGMA % 2 == 1 else ILLUM_SIGMA + 1
    illum = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    norm  = gray.astype(np.float32) / (illum + 1e-6) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def preprocess(bgr):
    L      = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L_norm = normalize_illumination(L)
    clahe  = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    L_eq   = clahe.apply(L_norm)
    return cv2.bilateralFilter(L_eq, d=9, sigmaColor=50, sigmaSpace=50)


# ===========================================================================
# SEGMENTASI JALAN
# ===========================================================================

def segment_road(bgr):
    """Isolasi jalan dengan menghapus langit dan vegetasi.

    Tahap 1: Mask langit (hue biru, separuh atas gambar)
    Tahap 2: Mask vegetasi/rumput (hue hijau, green > red)
    Tahap 3: Pilih komponen terbesar paling bawah = jalan
    Fallback: estimasi horizon Canny jika gagal
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    sky_mask         = np.zeros((h, w), dtype=np.uint8)
    top_h            = h // 2
    sky_region       = (
        (H[:top_h] >= SKY_HUE_LOW) & (H[:top_h] <= SKY_HUE_HIGH) &
        (S[:top_h] <= SKY_SAT_MAX) & (V[:top_h] >= SKY_VAL_MIN)
    ).astype(np.uint8) * 255
    sky_mask[:top_h] = sky_region

    g, r     = bgr[:, :, 1].astype(np.int16), bgr[:, :, 2].astype(np.int16)
    veg_mask = (
        (H >= VEG_HUE_LOW) & (H <= VEG_HUE_HIGH) &
        (S >= VEG_SAT_MIN) & (g > r)
    ).astype(np.uint8) * 255

    road_cand = cv2.bitwise_not(cv2.bitwise_or(sky_mask, veg_mask))
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROAD_MORPH_K, ROAD_MORPH_K))
    road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_CLOSE, k, iterations=2)
    road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_OPEN,  k, iterations=1)

    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(road_cand, connectivity=8)
    best_lbl, best_score = -1, -1.0
    for lbl in range(1, n_lbl):
        score = stats[lbl, cv2.CC_STAT_AREA] * (centroids[lbl][1] / h)
        if score > best_score:
            best_score, best_lbl = score, lbl

    if best_lbl == -1:
        gray_upper = cv2.cvtColor(bgr[:h // 2], cv2.COLOR_BGR2GRAY)
        row_sums   = np.sum(cv2.Canny(gray_upper, 50, 150), axis=1)
        horizon    = max(int(np.argmax(row_sums)), h // 10) if row_sums.max() > 0 else h // 3
        mask       = np.zeros((h, w), dtype=np.uint8)
        mask[horizon:] = 255
        return mask

    comp        = (labels == best_lbl).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled      = comp.copy()
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


# ===========================================================================
# EKSTRAKSI FITUR (9-channel, sinkron dengan model yang disimpan)
# ===========================================================================

def extract_features(bgr, prep):
    """9-channel feature map (H, W, 9), semua nilai dalam [0, 1].

    Ch 0: BGR Blue    Ch 3: HSV Hue   Ch 6: Gradient Mag
    Ch 1: BGR Green   Ch 4: HSV Sat   Ch 7: Blackhat
    Ch 2: BGR Red     Ch 5: HSV Val   Ch 8: LBP

    Blackhat dan LBP dihitung dari grayscale BGR ORIGINAL (bukan prep).
    """
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    b = bgr[:, :, 0].astype(np.float32) / 255.0
    g = bgr[:, :, 1].astype(np.float32) / 255.0
    r = bgr[:, :, 2].astype(np.float32) / 255.0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0] / 179.0
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0

    sx   = cv2.Sobel(prep, cv2.CV_32F, 1, 0, ksize=3)
    sy   = cv2.Sobel(prep, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(sx ** 2 + sy ** 2)
    grad = mag / mag.max() if mag.max() > 0 else mag

    bk_kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bk_raw   = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, bk_kern).astype(np.float32)
    bk_max   = bk_raw.max()
    blackhat = bk_raw / bk_max if bk_max > 0 else bk_raw

    lbp_raw = local_binary_pattern(gray_u8, P=8, R=1, method='uniform').astype(np.float32)
    lbp_max = lbp_raw.max()
    lbp     = lbp_raw / lbp_max if lbp_max > 0 else lbp_raw

    return np.stack([b, g, r, hue, sat, val, grad, blackhat, lbp], axis=-1).astype(np.float32)


# ===========================================================================
# SUPERPIXEL & POST-PROCESSING
# ===========================================================================

def compute_superpixels(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return slic(rgb, n_segments=SLIC_N_SEG, compactness=SLIC_COMPACT,
                sigma=1.0, start_label=0).astype(np.int32)


def aggregate_features(feat_map, segments):
    n_sp = segments.max() + 1
    n_ch = feat_map.shape[2]
    agg  = np.zeros((n_sp, n_ch), dtype=np.float32)
    for sp_id in range(n_sp):
        px = segments == sp_id
        if px.any():
            agg[sp_id] = feat_map[px].mean(axis=0)
    return agg


def postprocess(mask, min_area=None):
    min_area = min_area or MIN_AREA
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (POSTPROC_K, POSTPROC_K))
    cleaned  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cleaned  = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 255, cv2.FILLED)
    return result


# ===========================================================================
# TIGA MODEL DETEKSI
# ===========================================================================

def detect_baseline(bgr, min_area=MIN_AREA):
    road_mask = segment_road(bgr)
    prep      = preprocess(bgr)
    thresh    = cv2.adaptiveThreshold(
        prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8
    )
    pothole = cv2.bitwise_not(thresh)
    pothole = cv2.bitwise_and(pothole, road_mask)
    pothole[:int(bgr.shape[0] * TOP_CROP_PCT)] = 0
    return postprocess(pothole, min_area)


def detect_kmeans(bgr, min_area=MIN_AREA):
    road_mask = segment_road(bgr)
    h, w      = bgr.shape[:2]
    ws, hs    = int(w * WORK_SCALE), int(h * WORK_SCALE)
    bgr_s     = cv2.resize(bgr,       (ws, hs), interpolation=cv2.INTER_AREA)
    road_s    = cv2.resize(road_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)

    prep     = preprocess(bgr_s)
    pixels   = prep.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, K=3, bestLabels=None, criteria=criteria,
        attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )
    darkest = int(np.argmin(centers.flatten()))
    mask_s  = (labels.reshape(hs, ws) == darkest).astype(np.uint8) * 255
    mask_s  = cv2.bitwise_and(mask_s, road_s)

    pothole = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    pothole[:int(h * TOP_CROP_PCT)] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (POSTPROC_K, POSTPROC_K))
    pothole = cv2.morphologyEx(pothole, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(pothole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(pothole)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 255, cv2.FILLED)
    return result


def detect_rf(bgr, clf, min_area=MIN_AREA):
    road_mask = segment_road(bgr)
    h, w      = bgr.shape[:2]
    ws, hs    = int(w * WORK_SCALE), int(h * WORK_SCALE)
    bgr_s     = cv2.resize(bgr,       (ws, hs), interpolation=cv2.INTER_AREA)
    road_s    = cv2.resize(road_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)

    prep         = preprocess(bgr_s)
    feat_map     = extract_features(bgr_s, prep)
    segments     = compute_superpixels(bgr_s)
    sp_feats     = aggregate_features(feat_map, segments)
    pred_labels  = clf.predict(sp_feats)

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
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    pred = (pred_mask > MASK_THRESH).astype(np.uint8)
    gt   = (gt_mask   > MASK_THRESH).astype(np.uint8)
    eps  = 1e-8

    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    tn = float(np.logical_and(pred == 0, gt == 0).sum())

    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    prec   = tp / (tp + fp + eps)
    rec    = tp / (tp + fn + eps)
    return {
        'IoU Pothole': round(iou_fg,                         4),
        'mIoU':        round((iou_fg + iou_bg) / 2,         4),
        'Dice':        round(2*tp / (2*tp + fp + fn + eps),  4),
        'Pixel Acc':   round((tp+tn)/(tp+tn+fp+fn+eps),      4),
        'Precision':   round(prec,                            4),
        'Recall':      round(rec,                             4),
        'F1':          round(2*prec*rec/(prec+rec+eps),       4),
    }


# ===========================================================================
# HELPER
# ===========================================================================

def make_overlay(bgr, mask, color_bgr=(0, 0, 255), alpha=0.5):
    overlay = bgr.copy()
    colored = np.zeros_like(bgr)
    colored[:] = color_bgr
    px = mask > 0
    overlay[px] = (alpha * colored[px] + (1 - alpha) * bgr[px]).astype(np.uint8)
    return overlay


def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_uploaded_image(uploaded_file):
    return cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)


@st.cache_resource
def load_rf_model():
    """Load model RF-9ch dari disk. Di-cache agar tidak reload setiap interaksi."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ===========================================================================
# STREAMLIT UI
# ===========================================================================

def main():
    st.set_page_config(
        page_title='Pothole Detection',
        page_icon=None,
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.title('Pothole Detection')
    st.caption('Computer Vision Final Project — BINUS University Semester 4')
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('Pengaturan')

        method = st.radio(
            'Pilih Model',
            options=['Random Forest (9 fitur)', 'K-Means', 'Adaptive Threshold'],
            index=0,
        )

        st.divider()
        st.subheader('Parameter')
        min_area = st.slider(
            'Area minimum deteksi (px)',
            min_value=100, max_value=3000, value=500, step=100,
            help='Deteksi lebih kecil dari nilai ini dianggap noise dan dihapus.',
        )

        st.divider()
        with st.expander('Tentang setiap model'):
            st.markdown("""\
**Random Forest — 9 fitur (Eksperimen)**
Supervised — dilatih dari 398 gambar berlabel GT (split 80:20).
9 fitur per superpixel: BGR x3, HSV x3, Gradient, Blackhat, LBP.
Membutuhkan `model/model_rf_9ch.joblib`.

**K-Means (Advanced)**
Unsupervised — tidak perlu training.
K-Means (K=3) pada intensitas piksel.
Cluster paling gelap = kandidat lubang.

**Adaptive Threshold (Baseline)**
Tidak perlu training.
Threshold lokal berdasarkan intensitas piksel sekitar.
Paling cepat dan paling sederhana.""")

    # ── Load model RF jika dipilih ─────────────────────────────────────────
    clf = None
    if 'Random Forest' in method:
        clf = load_rf_model()
        if clf is None:
            st.error(
                f'Model tidak ditemukan: `{MODEL_PATH}`\n\n'
                'Latih model terlebih dahulu dengan menjalankan **Bagian 5** '
                'di `notebooks/main_training.ipynb`, lalu restart app ini.'
            )
            return
        if clf.n_features_in_ != N_FEATURES:
            st.error(
                f'**Model tidak kompatibel** — model lama menggunakan '
                f'**{clf.n_features_in_} fitur**, pipeline sekarang menggunakan '
                f'**{N_FEATURES} fitur**.'
            )
            st.info(
                '**Cara memperbaiki (jalankan notebook secara berurutan):**\n\n'
                '1. Buka `notebooks/main_training.ipynb`\n'
                '2. **Cell 2.3** — hapus CSV lama dulu: jalankan `SP_CSV.unlink()` '
                'di cell baru, lalu jalankan ulang cell 2.3 untuk rebuild CSV 9-fitur (~5–8 menit)\n'
                '3. **Cell 2.4** — jalankan ulang untuk rebuild sp_train.csv & sp_val.csv\n'
                '4. **Bagian 5** — jalankan cell training RF untuk simpan model baru\n'
                '5. Restart app ini (Ctrl+C lalu `streamlit run app/app.py`)'
            )
            return
        st.sidebar.success(f'Model: {clf.n_estimators} pohon, {clf.n_features_in_} fitur')

    # ── Upload gambar ──────────────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        st.subheader('Upload Gambar Jalan')
        uploaded = st.file_uploader(
            'Pilih gambar (JPG / PNG)',
            type=['jpg', 'jpeg', 'png'],
            label_visibility='collapsed',
        )

    with col_info:
        st.subheader('Informasi')
        if uploaded is not None:
            bgr = read_uploaded_image(uploaded)
            if bgr is None:
                st.error('Gambar tidak dapat dibaca.')
                return
            h, w = bgr.shape[:2]
            st.metric('Lebar',  f'{w} px')
            st.metric('Tinggi', f'{h} px')
            st.metric('Model',  method.split(' ')[0])
        else:
            st.info('Belum ada gambar.')
            return

    # ── Tombol deteksi ─────────────────────────────────────────────────────
    if not st.button('Deteksi Lubang', type='primary', use_container_width=True):
        st.image(bgr_to_rgb(bgr), caption='Gambar yang diupload', use_container_width=True)
        return

    t_start = time.perf_counter()
    with st.spinner('Memproses gambar...'):
        if 'Random Forest' in method:
            pred_mask = detect_rf(bgr, clf, min_area)
        elif 'K-Means' in method:
            pred_mask = detect_kmeans(bgr, min_area)
        else:
            pred_mask = detect_baseline(bgr, min_area)
        road_mask = segment_road(bgr)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── Tampilkan hasil ────────────────────────────────────────────────────
    st.subheader('Hasil Deteksi')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(bgr_to_rgb(bgr), caption='Gambar Asli', use_container_width=True)
    with col2:
        road_overlay = make_overlay(bgr, road_mask, color_bgr=(0, 180, 0), alpha=0.4)
        st.image(bgr_to_rgb(road_overlay), caption='Area Jalan (hijau)', use_container_width=True)
    with col3:
        pred_overlay = make_overlay(bgr, pred_mask, color_bgr=(0, 0, 220), alpha=0.55)
        st.image(bgr_to_rgb(pred_overlay), caption='Prediksi Lubang (biru)', use_container_width=True)

    pct = (pred_mask > MASK_THRESH).mean() * 100
    m1, m2 = st.columns(2)
    m1.info(f'Area lubang terdeteksi: **{pct:.2f}%** dari total gambar')
    m2.info(f'Waktu prediksi ({method.split(" ")[0]}): **{elapsed_ms:.1f} ms**')

    ok, buf = cv2.imencode('.png', pred_mask)
    if ok:
        st.download_button(
            'Unduh Predicted Mask (PNG)',
            data=buf.tobytes(),
            file_name='predicted_mask.png',
            mime='image/png',
        )

    # ── Evaluasi dengan Ground Truth (opsional) ────────────────────────────
    st.divider()
    st.subheader('Evaluasi dengan Ground Truth (Opsional)')
    st.caption(
        'Upload mask ground truth untuk menghitung metrik. '
        'Format: **grayscale PNG** — piksel putih (255) = lubang, hitam (0) = jalan.'
    )

    uploaded_gt = st.file_uploader(
        'Upload ground truth mask (PNG grayscale)',
        type=['png', 'jpg'],
        key='gt_uploader',
        label_visibility='collapsed',
    )

    if uploaded_gt is not None:
        gt_raw = cv2.imdecode(
            np.frombuffer(uploaded_gt.read(), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        if gt_raw is None:
            st.error('Tidak dapat membaca file mask. Pastikan file adalah gambar grayscale yang valid.')
            return

        # Resize GT ke ukuran pred_mask jika berbeda
        if gt_raw.shape != pred_mask.shape:
            gt_raw = cv2.resize(
                gt_raw, (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Binarisasi ketat: nilai > 127 → 255 (putih), lainnya → 0 (hitam)
        gt_bin = ((gt_raw > MASK_THRESH).astype(np.uint8)) * 255

        n_pothole_px = int((gt_bin > 0).sum())
        n_total_px   = gt_bin.size
        gt_pct       = n_pothole_px / n_total_px * 100

        # Tampilkan preview GT setelah binarisasi
        st.markdown('**Preview Ground Truth (setelah binarisasi >127):**')
        cgt1, cgt2, cgt3 = st.columns([1, 1, 2])
        with cgt1:
            st.image(gt_bin, caption='GT Mask (biner)', use_container_width=True, clamp=True)
        with cgt2:
            st.metric('Piksel Lubang', f'{n_pothole_px:,}')
            st.metric('Persentase',    f'{gt_pct:.2f}%')
        with cgt3:
            st.info(
                'Mask sudah dibinarisasi: putih = lubang, hitam = jalan.\n\n'
                'Jika preview tidak sesuai, pastikan mask asli menggunakan '
                'nilai 255 (putih) untuk area lubang dan 0 (hitam) untuk jalan.'
            )

        if n_pothole_px == 0:
            st.warning('Ground truth mask tampak kosong (tidak ada piksel putih). Periksa format file.')
            return

        metrics = compute_metrics(pred_mask, gt_bin)

        st.markdown('**Metrik Evaluasi:**')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('mIoU',        f"{metrics['mIoU']:.4f}",        help='Metrik utama — target >= 0.60')
        c2.metric('IoU Pothole', f"{metrics['IoU Pothole']:.4f}", help='IoU kelas lubang saja')
        c3.metric('Dice',        f"{metrics['Dice']:.4f}")
        c4.metric('Pixel Acc',   f"{metrics['Pixel Acc']:.4f}")

        c5, c6, c7 = st.columns(3)
        c5.metric('Precision', f"{metrics['Precision']:.4f}")
        c6.metric('Recall',    f"{metrics['Recall']:.4f}")
        c7.metric('F1',        f"{metrics['F1']:.4f}")

        miou = metrics['mIoU']
        if miou >= 0.60:
            st.success(f'mIoU = {miou:.4f} — target 0.60 tercapai.')
        elif miou >= 0.40:
            st.warning(f'mIoU = {miou:.4f} — mendekati target, belum tercapai.')
        else:
            st.error(f'mIoU = {miou:.4f} — jauh dari target 0.60.')

        st.subheader('Perbandingan Prediksi vs Ground Truth')
        ca, cb = st.columns(2)
        with ca:
            gt_overlay = make_overlay(bgr, gt_bin, color_bgr=(0, 200, 0), alpha=0.5)
            st.image(bgr_to_rgb(gt_overlay), caption='Ground Truth (hijau)', use_container_width=True)
        with cb:
            st.image(bgr_to_rgb(pred_overlay), caption='Prediksi (biru)', use_container_width=True)


if __name__ == '__main__':
    main()
