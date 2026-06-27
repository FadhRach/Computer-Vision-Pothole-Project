"""
Pothole Detection — Streamlit Web App
Self-contained: every pipeline function lives in this single file (ready for
Streamlit Cloud deployment).

Two pages (sidebar navigation):
    - Detection : upload image -> (optional) GT mask -> run 3 models -> compare
    - Model Info: how each model works, strengths/weaknesses, ideal conditions, experiment

Run from the project root:
    python -m streamlit run app/app.py

Main supervised model: model/model_rf_9ch.joblib (Random Forest, 9 features).
"""

import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern


# ===========================================================================
# CONFIGURATION
# ===========================================================================

ROOT     = Path(__file__).resolve().parent.parent
RF_PATH  = ROOT / 'model' / 'model_rf_9ch.joblib'

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

# Road segmentation
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
N_FEATURES   = 9
TARGET_MIOU  = 0.60

# Overlay color per model (BGR)
COLOR_AT   = (40, 140, 255)   # orange
COLOR_KM   = (200, 120, 0)    # blue
COLOR_RF   = (0, 0, 220)      # red
COLOR_GT   = (0, 200, 0)      # green

# Experiment results (mean mIoU on the same 100 validation images)
RF_MIOU       = 0.5179
CATBOOST_MIOU = 0.5440        # calibrated CatBoost (threshold 0.70)
EXPERIMENT_TABLE = pd.DataFrame(
    {'mIoU': [0.4028, 0.4442, RF_MIOU, CATBOOST_MIOU]},
    index=['Adaptive Threshold', 'K-Means', 'Random Forest (used)',
           'CatBoost (experiment, thr 0.70)'],
)


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
# ROAD SEGMENTATION
# ===========================================================================

def segment_road(bgr):
    """Isolate the road by removing sky and vegetation.

    Step 1: Sky mask (blue hue, upper half of the image)
    Step 2: Vegetation/grass mask (green hue, green > red)
    Step 3: Pick the largest bottom-weighted component = road
    Fallback: Canny horizon estimate if everything fails
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
# FEATURE EXTRACTION (9-channel, in sync with the saved model)
# ===========================================================================

def extract_features(bgr, prep):
    """9-channel feature map (H, W, 9), all values in [0, 1].

    Ch 0: BGR Blue    Ch 3: HSV Hue   Ch 6: Gradient Mag
    Ch 1: BGR Green   Ch 4: HSV Sat   Ch 7: Blackhat
    Ch 2: BGR Red     Ch 5: HSV Val   Ch 8: LBP

    Blackhat and LBP are computed from the ORIGINAL BGR grayscale (not prep),
    because illumination normalization suppresses the dark-spot signal Blackhat needs.
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
# SUPERPIXELS & POST-PROCESSING
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
# THREE DETECTION MODELS
# ===========================================================================

def detect_baseline(bgr, min_area=MIN_AREA):
    """Adaptive Threshold: locally dark pixels on the road area = pothole."""
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
    """K-Means (K=3) on intensity; the darkest cluster = pothole candidate."""
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
    """Random Forest on the 9 features per SLIC superpixel."""
    road_mask = segment_road(bgr)
    h, w      = bgr.shape[:2]
    ws, hs    = int(w * WORK_SCALE), int(h * WORK_SCALE)
    bgr_s     = cv2.resize(bgr,       (ws, hs), interpolation=cv2.INTER_AREA)
    road_s    = cv2.resize(road_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)

    prep        = preprocess(bgr_s)
    feat_map    = extract_features(bgr_s, prep)
    segments    = compute_superpixels(bgr_s)
    sp_feats    = aggregate_features(feat_map, segments)
    pred_labels = clf.predict(sp_feats)

    pothole_segs = np.where(pred_labels == 1)[0]
    mask_s       = np.isin(segments, pothole_segs).astype(np.uint8) * 255
    mask_s       = cv2.bitwise_and(mask_s, road_s)

    pothole = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    pothole[:int(h * TOP_CROP_PCT)] = 0
    return postprocess(pothole, min_area)


# ===========================================================================
# EVALUATION METRICS
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
# HELPERS
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


def hex_color(color_bgr):
    b, g, r = color_bgr
    return f'#{r:02x}{g:02x}{b:02x}'


def read_uploaded_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detected_area_pct(mask):
    return (mask > MASK_THRESH).mean() * 100


@st.cache_resource(show_spinner=False)
def load_rf_model():
    """Load the 9-channel Random Forest. Cached so it is not reloaded every interaction.

    Returns (model, error_message). model is None on failure / missing file.
    """
    if not RF_PATH.exists():
        return None, f'Model file not found: {RF_PATH.name}'
    try:
        clf = joblib.load(RF_PATH)
    except Exception as exc:
        return None, f'Failed to load model: {exc}'
    if getattr(clf, 'n_features_in_', None) != N_FEATURES:
        return None, (f'Incompatible model ({getattr(clf, "n_features_in_", "?")} features, '
                      f'expected {N_FEATURES}).')
    return clf, None


# ===========================================================================
# STYLING
# ===========================================================================

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; max-width: 1280px; }

    .hero {
        background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.8rem 2rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing:-0.5px; }
    .hero p  { margin: 0.35rem 0 0 0; opacity: 0.88; font-size: 0.98rem; }
    .hero .chips { margin-top: 0.9rem; }
    .chip {
        display:inline-block; background: rgba(255,255,255,0.14); color:#fff;
        padding: 0.28rem 0.7rem; border-radius: 999px; font-size: 0.78rem;
        margin-right: 0.4rem; margin-bottom: 0.3rem; border: 1px solid rgba(255,255,255,0.22);
        font-weight:600;
    }
    .chip.target { background:#1b5e20; border-color:#2e7d32; }

    .step-badge {
        display:inline-flex; align-items:center; justify-content:center;
        width: 26px; height: 26px; border-radius: 50%; background:#2c5364; color:#fff;
        font-weight:700; font-size:0.85rem; margin-right:0.55rem;
    }
    .step-title { font-size:1.15rem; font-weight:700; margin: 0.4rem 0 0.6rem 0;
                  display:flex; align-items:center; }

    .model-tag {
        display:inline-block; padding:0.2rem 0.6rem; border-radius:8px; font-weight:700;
        font-size:0.82rem; color:#fff;
    }
    .flow-step {
        display:inline-block; background:#eef2f6; color:#2c3e50; border:1px solid #d6dee6;
        padding:0.3rem 0.6rem; border-radius:8px; font-size:0.8rem; margin:0.15rem 0.15rem;
        font-weight:600;
    }
    .flow-arrow { color:#90a4ae; font-weight:700; margin:0 0.05rem; }

    .pro  { color:#16a34a; font-weight:700; }
    .con  { color:#dc2626; font-weight:700; }
    /* text color forced dark so it stays readable in light & dark mode
       (the box background is always light green) */
    .ideal-box {
        background:#e8f5e9; border-left:4px solid #2e7d32; border-radius:8px;
        padding:0.6rem 0.8rem; margin-top:0.6rem; font-size:0.9rem; color:#14532d;
    }
    .ideal-box b { color:#14532d; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }

    .status-row { display:flex; align-items:center; font-size:0.9rem; margin:0.3rem 0; }
    .dot {
        display:inline-block; width:11px; height:11px; border-radius:50%;
        margin-right:0.55rem; flex:0 0 auto;
    }
    .dot.green { background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,0.18); }
    .dot.red   { background:#ef4444; box-shadow:0 0 0 3px rgba(239,68,68,0.18); }
    .status-detail { color:#64748b; font-size:0.78rem; margin-left:0.4rem; }
</style>
"""


# ===========================================================================
# UI COMPONENTS
# ===========================================================================

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>Road Pothole Detection</h1>
            <p>Traditional Computer Vision (no deep learning) · BINUS University, Semester 4</p>
            <div class="chips">
                <span class="chip">Adaptive Threshold</span>
                <span class="chip">K-Means Clustering</span>
                <span class="chip">Random Forest + SLIC</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_header(num, title):
    st.markdown(
        f'<div class="step-title"><span class="step-badge">{num}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def flow_html(steps):
    inner = '<span class="flow-arrow">&rarr;</span>'.join(
        f'<span class="flow-step">{s}</span>' for s in steps
    )
    return f'<div style="margin:0.4rem 0">{inner}</div>'


def status_dot(label, ok, detail=''):
    """Model status indicator: green dot (ready) / red dot (not available)."""
    cls   = 'green' if ok else 'red'
    extra = f'<span class="status-detail">{detail}</span>' if detail else ''
    st.markdown(
        f'<div class="status-row"><span class="dot {cls}"></span><span>{label}</span>{extra}</div>',
        unsafe_allow_html=True,
    )


def render_model_card(model, pred_mask, bgr, elapsed_ms, gt_bin):
    """One detection result card for a single model."""
    with st.container(border=True):
        st.markdown(
            f'<span class="model-tag" style="background:{hex_color(model["color"])}">'
            f'{model["name"]}</span>',
            unsafe_allow_html=True,
        )
        overlay = make_overlay(bgr, pred_mask, color_bgr=model['color'], alpha=0.55)
        st.image(bgr_to_rgb(overlay), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric('Pothole area', f'{detected_area_pct(pred_mask):.2f}%')
        c2.metric('Time', f'{elapsed_ms:.0f} ms')

        if gt_bin is not None:
            m = compute_metrics(pred_mask, gt_bin)
            mc1, mc2 = st.columns(2)
            mc1.metric('mIoU', f"{m['mIoU']:.4f}")
            mc2.metric('IoU Pothole', f"{m['IoU Pothole']:.4f}")
            st.caption(
                f"Precision {m['Precision']:.3f} · Recall {m['Recall']:.3f} · F1 {m['F1']:.3f}"
            )

        ok, buf = cv2.imencode('.png', pred_mask)
        if ok:
            st.download_button(
                'Download mask (PNG)', data=buf.tobytes(),
                file_name=f'pred_{model["key"]}.png', mime='image/png',
                use_container_width=True, key=f'dl_{model["key"]}',
            )


# Per-model characteristics documentation
MODEL_DOCS = [
    {
        'name': 'Adaptive Threshold', 'color': COLOR_AT, 'badge': 'Baseline',
        'flow': ['BGR', 'Grayscale', 'Adaptive Threshold (local)', 'Inverse', 'Prediction'],
        'how': 'Converts the image to grayscale, then compares each pixel against the average '
               'brightness of its neighbours. Pixels darker than their surroundings are treated '
               'as potholes (inverted). In short, it only looks for neighbouring pixels that are '
               'similar and dark to label them as a pothole.',
        'pros': ['Very fast and lightweight', 'No training / labelled data needed',
                 'Simple and easy to understand'],
        'cons': ['Easily fooled by shadows & dark stains', 'No understanding of shape or texture',
                 'Sensitive to uneven lighting'],
        'ideal': 'Good for clean road images with dark potholes that strongly contrast against '
                 'a bright road surface.',
    },
    {
        'name': 'K-Means Clustering', 'color': COLOR_KM, 'badge': 'Advanced',
        'flow': ['Resize 0.5x', 'K-Means (K=3) on intensity', 'Darkest cluster',
                 'Restrict to road area', 'Prediction'],
        'how': 'Groups pixel intensities into 3 clusters without labels (unsupervised). The '
               'cluster with the darkest mean intensity is taken as the pothole candidate, then '
               'restricted to the road area only.',
        'pros': ['Adapts to each image contrast', 'Unsupervised (no labels)',
                 'Handles dark/water variations better than a fixed threshold'],
        'cons': ['Still relies only on darkness', 'Large shadows can be detected too',
                 'Does not use texture or shape'],
        'ideal': 'Good for road images with water puddles or potholes with mixed conditions — '
                 'normal asphalt, pothole edges, and dark/water areas — because it can capture '
                 'varying dark regions.',
    },
    {
        'name': 'Random Forest + SLIC', 'color': COLOR_RF, 'badge': 'Supervised (main)',
        'flow': ['SLIC superpixels (~250)', 'Extract 9 features/superpixel',
                 'Random Forest classification', 'Merge pothole superpixels', 'Prediction'],
        'how': 'Splits the image into SLIC superpixels, then represents each superpixel with 9 '
               'features (BGR, HSV, Gradient, Blackhat, LBP). A Random Forest (200 trees) trained '
               'on 398 labelled images decides whether each superpixel is a pothole or not. '
               'Because it uses colour, texture, and shape together, it is far more accurate.',
        'pros': ['Combines colour + texture + shape', 'Highest accuracy (mIoU)',
                 'Good for cracked roads & diverse conditions'],
        'cons': ['Needs training & labelled data', 'Heavier than the other two models',
                 'On "spontaneous" potholes (e.g. coloured water puddles with no pattern), the '
                 'pothole centre is sometimes treated as non-pothole'],
        'ideal': 'Good for cracked roads and diverse surface conditions — it captures more road '
                 'variations and is much better overall.',
    },
]


def render_model_characteristics():
    step_header('?', 'How Each Model Works & Characteristics')
    st.markdown(
        'All three models share the **same pre-processing pipeline**, and differ only at the '
        'decision-making stage.'
    )
    st.markdown('**Shared pipeline:**', help='Applies to all models.')
    st.markdown(
        flow_html(['Illumination normalization', 'CLAHE + bilateral filter',
                   'Road segmentation (remove sky & vegetation)', 'Crop top 10%',
                   'Morphological post-processing + area filter']),
        unsafe_allow_html=True,
    )

    tabs = st.tabs([d['name'] for d in MODEL_DOCS])
    for tab, doc in zip(tabs, MODEL_DOCS):
        with tab:
            st.markdown(
                f'<span class="model-tag" style="background:{hex_color(doc["color"])}">'
                f'{doc["name"]}</span> &nbsp;<small>{doc["badge"]}</small>',
                unsafe_allow_html=True,
            )
            st.markdown(flow_html(doc['flow']), unsafe_allow_html=True)
            st.write(doc['how'])
            col_pro, col_con = st.columns(2)
            with col_pro:
                st.markdown('<span class="pro">Strengths</span>', unsafe_allow_html=True)
                for p in doc['pros']:
                    st.markdown(f'- {p}')
            with col_con:
                st.markdown('<span class="con">Weaknesses</span>', unsafe_allow_html=True)
                for c in doc['cons']:
                    st.markdown(f'- {c}')
            st.markdown(
                f'<div class="ideal-box"><b>Ideal condition:</b> {doc["ideal"]}</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        '9 features per superpixel: ' + ', '.join(FEATURE_NAMES) +
        '. Blackhat highlights small dark spots, LBP captures local texture.'
    )


def render_experiment():
    step_header('!', 'Experiment: Why Keep Random Forest?')
    st.markdown(
        'We tested modern boosting classifiers (**XGBoost, LightGBM, CatBoost**) on the '
        '**same 9 features and pipeline**. At the default threshold (0.5), all three were '
        'actually slightly below Random Forest. After *decision threshold* calibration, '
        f'**CatBoost (threshold 0.70)** reached mIoU **{CATBOOST_MIOU:.4f}** — only a small '
        f'**(+{CATBOOST_MIOU - RF_MIOU:.4f})** gain over Random Forest ({RF_MIOU:.4f}).'
    )
    col_tab, col_chart = st.columns([1, 1])
    with col_tab:
        styled = (EXPERIMENT_TABLE.style
                  .format('{:.4f}')
                  .highlight_max(subset=['mIoU'], color='#c8e6c9'))
        st.dataframe(styled, use_container_width=True)
    with col_chart:
        st.bar_chart(EXPERIMENT_TABLE, horizontal=True, color=hex_color(COLOR_RF))
    st.info(
        'Because the gain is small while Random Forest is simpler, lighter, and easier to '
        'explain (feature importance), **Random Forest was chosen as the main model** of this '
        'app. Full experiment details are in `notebooks/experimentensemble_training.ipynb`.'
    )


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar():
    """Model status (dot indicator) + settings. Used on every page."""
    clf, _ = load_rf_model()
    rf_ready = clf is not None
    with st.sidebar:
        st.subheader('Model Status')
        status_dot('Adaptive Threshold', True, 'ready')
        status_dot('K-Means', True, 'ready')
        if rf_ready:
            status_dot('Random Forest', True, f'{clf.n_estimators} trees')
        else:
            status_dot('Random Forest', False, 'not available')

        st.divider()
        st.subheader('Settings')
        st.slider(
            'Minimum detection area (px)',
            min_value=100, max_value=3000, value=MIN_AREA, step=100, key='min_area',
            help='Detections smaller than this value are treated as noise and removed.',
        )

        st.divider()
        st.caption('Use the navigation menu above: **Detection** to process an image, '
                   '**Model Info** for an explanation of each model.')


# ===========================================================================
# PAGE: DETECTION
# ===========================================================================

def page_detect():
    clf, rf_error = load_rf_model()
    rf_ready = clf is not None
    min_area = st.session_state.get('min_area', MIN_AREA)

    if not rf_ready:
        st.warning(
            f'Random Forest skipped — {rf_error} '
            'The app still runs with Adaptive Threshold & K-Means. To enable RF, provide '
            '`model/model_rf_9ch.joblib` (train it in `notebooks/main_training.ipynb`).'
        )

    # -- Step 1 -- Upload image --------------------------------------------
    step_header(1, 'Upload Road Image')
    uploaded = st.file_uploader(
        'Choose a road image (JPG / PNG)', type=['jpg', 'jpeg', 'png'],
        label_visibility='collapsed',
    )
    if uploaded is None:
        st.info('Please upload a road image to start detection. How each model works is '
                'explained on the **Model Info** page.')
        return

    bgr = read_uploaded_image(uploaded)
    if bgr is None:
        st.error('The image could not be read. Try another file (JPG/PNG).')
        return

    h, w = bgr.shape[:2]
    prev_col, info_col = st.columns([2, 1])
    with prev_col:
        st.image(bgr_to_rgb(bgr), caption='Uploaded image', use_container_width=True)
    with info_col:
        st.metric('Width', f'{w} px')
        st.metric('Height', f'{h} px')
        st.metric('Active models', f'{2 + int(rf_ready)}')

    # -- Step 2 -- Optional ground truth -----------------------------------
    st.write('')
    step_header(2, 'Upload Ground Truth Mask (Optional)')
    st.caption(
        'Used to score the mIoU of each model. Format: grayscale PNG — white (255) = pothole, '
        'black (0) = road. Skip this if you only want the predictions.'
    )
    uploaded_gt = st.file_uploader(
        'Upload ground truth mask', type=['png', 'jpg', 'jpeg'],
        key='gt_uploader', label_visibility='collapsed',
    )

    gt_bin = None
    if uploaded_gt is not None:
        gt_raw = cv2.imdecode(np.frombuffer(uploaded_gt.getvalue(), dtype=np.uint8),
                              cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            st.error('Could not read the mask. Make sure it is a valid grayscale image.')
        else:
            if gt_raw.shape != (h, w):
                gt_raw = cv2.resize(gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_candidate = ((gt_raw > MASK_THRESH).astype(np.uint8)) * 255
            n_pothole = int((gt_candidate > 0).sum())
            g1, g2 = st.columns([1, 2])
            with g1:
                gt_overlay = make_overlay(bgr, gt_candidate, color_bgr=COLOR_GT, alpha=0.5)
                st.image(bgr_to_rgb(gt_overlay), caption='Ground Truth (green)',
                         use_container_width=True)
            with g2:
                st.metric('Pothole pixels', f'{n_pothole:,}')
                st.metric('Percentage', f'{n_pothole / gt_candidate.size * 100:.2f}%')
                if n_pothole == 0:
                    st.warning('The mask looks empty (no white pixels). Check the file format.')
            if n_pothole > 0:
                gt_bin = gt_candidate

    # -- Step 3 -- Run detection -------------------------------------------
    st.write('')
    step_header(3, 'Run Detection')

    models = [
        {'key': 'at', 'name': 'Adaptive Threshold', 'color': COLOR_AT,
         'fn': lambda b: detect_baseline(b, min_area)},
        {'key': 'km', 'name': 'K-Means', 'color': COLOR_KM,
         'fn': lambda b: detect_kmeans(b, min_area)},
    ]
    if rf_ready:
        models.append({'key': 'rf', 'name': 'Random Forest', 'color': COLOR_RF,
                       'fn': lambda b: detect_rf(b, clf, min_area)})

    if not st.button('Run All Models', type='primary', use_container_width=True):
        return

    results = []
    progress = st.progress(0.0, text='Processing...')
    for i, model in enumerate(models):
        try:
            t0 = time.perf_counter()
            pred = model['fn'](bgr)
            elapsed = (time.perf_counter() - t0) * 1000
            results.append({'model': model, 'mask': pred, 'ms': elapsed,
                            'metrics': compute_metrics(pred, gt_bin) if gt_bin is not None else None})
        except Exception as exc:  # one failing model must not break the page
            st.error(f"Model {model['name']} failed: {exc}")
        progress.progress((i + 1) / len(models), text=f"Done: {model['name']}")
    progress.empty()

    if not results:
        st.error('No model ran successfully.')
        return

    # -- Per-model results -------------------------------------------------
    st.subheader('Detection Results')
    cols = st.columns(len(results))
    for col, res in zip(cols, results):
        with col:
            render_model_card(res['model'], res['mask'], bgr, res['ms'], gt_bin)

    # -- Comparison (when GT is provided) ----------------------------------
    if gt_bin is not None:
        st.subheader('Model Comparison')
        metric_cols = ['mIoU', 'IoU Pothole', 'Dice', 'Precision', 'Recall', 'F1']
        table = pd.DataFrame(
            {res['model']['name']: [res['metrics'][c] for c in metric_cols] for res in results},
            index=metric_cols,
        ).T.sort_values('mIoU', ascending=False)

        best_name = table.index[0]
        best_miou = table.iloc[0]['mIoU']
        verdict = ('target reached.' if best_miou >= TARGET_MIOU
                   else f'below the target of {TARGET_MIOU:.2f}.')
        st.markdown(f'Best model on this image: **{best_name}** '
                    f'with mIoU **{best_miou:.4f}** — {verdict}')
        st.dataframe(
            table.style.format('{:.4f}').highlight_max(subset=['mIoU'], color='#c8e6c9'),
            use_container_width=True,
        )
        st.bar_chart(table[['mIoU']], horizontal=True, color=hex_color(COLOR_RF))
    else:
        st.info('Upload a ground truth mask in Step 2 to see the mIoU metrics & '
                'the model comparison table.')


# ===========================================================================
# PAGE: MODEL INFO
# ===========================================================================

def page_info():
    render_model_characteristics()
    st.divider()
    render_experiment()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    st.set_page_config(
        page_title='Pothole Detection',
        page_icon=None,
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    pages = [
        st.Page(page_detect, title='Detection', icon=':material/search:', default=True),
        st.Page(page_info,   title='Model Info', icon=':material/menu_book:'),
    ]
    nav = st.navigation(pages, position='sidebar')
    render_sidebar()
    nav.run()


if __name__ == '__main__':
    main()
