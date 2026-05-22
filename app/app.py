"""
Pothole Detection Web Application
Run with: streamlit run app/app.py  (from project root)

Requires model/model_rf.joblib to be present.
Train it first by running notebooks/main_training.ipynb Cell 6.
"""

import sys
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

# Allow importing pipeline.py from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import (
    Config,
    run_baseline,
    run_kmeans,
    run_rf,
    load_model,
    compute_metrics,
    make_overlay,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model_rf.joblib"


# -----------------------------------------------------------------------
# Model loader
# -----------------------------------------------------------------------

@st.cache_resource
def get_rf_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(str(MODEL_PATH))


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def mask_to_bytes(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask)
    return buf.tobytes() if ok else b""


def build_config_from_sidebar() -> Config:
    cfg = Config()
    st.sidebar.subheader("Processing Settings")
    cfg.WORK_SCALE = st.sidebar.slider(
        "Work Scale", 0.25, 1.0, 0.5, 0.05,
        help="Downscale factor. Lower = faster, less precise."
    )
    cfg.POSTPROC_MIN_CONTOUR_AREA = st.sidebar.slider(
        "Min Detection Area (px)", 100, 5000, 500, 100,
        help="Regions smaller than this are removed as noise."
    )
    return cfg


# -----------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------

def display_results(bgr, road_mask, pred_mask, road_valid):
    overlay      = make_overlay(bgr, pred_mask, color_bgr=(0, 0, 255), alpha=0.5)
    road_overlay = make_overlay(bgr, road_mask, color_bgr=(0, 255, 0), alpha=0.35)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(bgr_to_pil(bgr), caption="Original Image", use_container_width=True)
    with col2:
        st.image(bgr_to_pil(road_overlay), caption="Road Segmentation (green)", use_container_width=True)
    with col3:
        st.image(bgr_to_pil(overlay), caption="Detected Potholes (red)", use_container_width=True)

    if not road_valid:
        st.warning(
            "Road segmentation used a fallback method. "
            "Detection may be less accurate on this image."
        )


def display_metrics(pred_mask, gt_mask):
    if gt_mask.shape != pred_mask.shape:
        gt_mask = cv2.resize(
            gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    m = compute_metrics(pred_mask, gt_mask)

    st.subheader("Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("mIoU",             f"{m['mIoU']:.4f}",          help="Mean IoU (foreground + background)")
    c2.metric("Dice Coefficient", f"{m['Dice']:.4f}",           help="2TP / (2TP + FP + FN)")
    c3.metric("Pixel Accuracy",   f"{m['PixelAccuracy']:.4f}",  help="Fraction of correctly classified pixels")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{m['Precision']:.4f}")
    c5.metric("Recall",    f"{m['Recall']:.4f}")
    c6.metric("F1",        f"{m['F1']:.4f}")

    if m["mIoU"] >= 0.6:
        st.success(f"mIoU = {m['mIoU']:.4f} — above the project target of 0.60")
    else:
        st.info(f"mIoU = {m['mIoU']:.4f} — project target is 0.60")


# -----------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------

METHODS = {
    "Eksperimen: Random Forest + SLIC": "rf",
    "Proposal Advanced: K-Means Clustering": "kmeans",
    "Proposal Baseline: Adaptive Thresholding": "baseline",
}

METHOD_INFO = {
    "rf": (
        "Supervised Random Forest trained on 498 labeled road images. "
        "Menggunakan 14-channel feature (7 grayscale texture/edge + 3 HSV color + "
        "4 edge/corner). Membutuhkan file model/model_rf.joblib."
    ),
    "kmeans": (
        "K-Means pixel clustering (k=3) pada ruang fitur gabungan. "
        "Cluster paling gelap diasumsikan sebagai pothole. Metode unsupervised — "
        "tidak membutuhkan training. Sesuai proposal asli."
    ),
    "baseline": (
        "Adaptive Gaussian Thresholding pada gambar yang sudah dikoreksi iluminasi. "
        "Cepat, deterministik, tidak membutuhkan file model."
    ),
}


def main():
    st.set_page_config(page_title="Pothole Detection", layout="wide")
    st.title("Deteksi Jalan Berlubang (Pothole Detection)")
    st.caption("Traditional Computer Vision Pipeline — BINUS University Final Project")

    st.sidebar.title("Pengaturan")
    method_label = st.sidebar.radio("Metode Deteksi", list(METHODS.keys()))
    method = METHODS[method_label]

    st.sidebar.info(METHOD_INFO[method])
    cfg = build_config_from_sidebar()

    # Load RF model if needed
    clf = None
    if method == "rf":
        clf = get_rf_model()
        if clf is None:
            st.error(
                f"File model tidak ditemukan: {MODEL_PATH}\n\n"
                "Latih model terlebih dahulu dengan menjalankan Cell 6 "
                "di notebooks/main_training.ipynb, lalu restart aplikasi ini."
            )
            return

    st.subheader("Upload Gambar Jalan")
    uploaded_img = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_img is None:
        st.info("Upload gambar jalan untuk memulai deteksi.")
        return

    bgr = load_image(uploaded_img)
    if bgr is None:
        st.error("Tidak dapat membaca gambar yang diupload.")
        return

    if st.button("Deteksi Pothole", type="primary"):
        with st.spinner("Menjalankan deteksi..."):
            if method == "rf":
                pred_mask, road_mask, road_valid = run_rf(bgr, clf, cfg)
            elif method == "kmeans":
                pred_mask, road_mask, road_valid = run_kmeans(bgr, cfg)
            else:
                pred_mask, road_mask, road_valid = run_baseline(bgr, cfg)

        st.subheader("Hasil Deteksi")
        display_results(bgr, road_mask, pred_mask, road_valid)

        st.download_button(
            "Download Predicted Mask (PNG)",
            data=mask_to_bytes(pred_mask),
            file_name="predicted_mask.png",
            mime="image/png",
        )

        st.subheader("Evaluasi dengan Ground Truth (opsional)")
        uploaded_gt = st.file_uploader(
            "Upload ground truth mask (grayscale PNG)", type=["png", "jpg"], key="gt"
        )
        if uploaded_gt is not None:
            gt_bytes = np.frombuffer(uploaded_gt.read(), dtype=np.uint8)
            gt_mask = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                display_metrics(pred_mask, gt_mask)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(Image.fromarray(gt_mask), caption="Ground Truth Mask", use_container_width=True)
                with col_b:
                    gt_overlay = make_overlay(bgr, gt_mask, color_bgr=(0, 255, 255), alpha=0.5)
                    st.image(bgr_to_pil(gt_overlay), caption="Ground Truth Overlay (cyan)", use_container_width=True)
            else:
                st.error("Tidak dapat membaca ground truth mask.")


if __name__ == "__main__":
    main()
