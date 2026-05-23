"""
Pothole Detection Web Application
Run with: streamlit run app.py

Requires model_rf.joblib to be present (train it first via main_training.ipynb Cell 6).
"""

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

from app.pipeline import (
    Config,
    run_baseline,
    run_rf,
    load_model,
    compute_metrics,
    make_overlay,
)


# -----------------------------------------------------------------------
# Model loader (cached so it only loads once per session)
# -----------------------------------------------------------------------

@st.cache_resource
def get_rf_model():
    model_path = "model_rf.joblib"
    if not Path(model_path).exists():
        return None
    return load_model(model_path)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_image(uploaded_file) -> np.ndarray:
    """Convert a Streamlit UploadedFile to a BGR numpy array."""
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
    c1.metric("mIoU",          f"{m['mIoU']:.4f}",          help="Mean IoU (foreground + background classes)")
    c2.metric("Dice Coefficient", f"{m['Dice']:.4f}",        help="2TP / (2TP + FP + FN)")
    c3.metric("Pixel Accuracy", f"{m['PixelAccuracy']:.4f}", help="Fraction of correctly classified pixels")

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

def main():
    st.set_page_config(page_title="Pothole Detection", layout="wide")
    st.title("Pothole Detection")
    st.caption("Traditional CV pipeline — Adaptive Thresholding vs Random Forest + SLIC")

    # Sidebar
    st.sidebar.title("Settings")
    method = st.sidebar.radio(
        "Detection Method",
        ["Advanced (Random Forest)", "Baseline (Adaptive Threshold)"],
        help="RF model must be trained first via main_training.ipynb."
    )

    method_info = {
        "Baseline (Adaptive Threshold)": (
            "Applies local Gaussian thresholding to the illumination-corrected image. "
            "Fast and deterministic. No model file required."
        ),
        "Advanced (Random Forest)": (
            "Superpixel (SLIC) features classified by a Random Forest trained on "
            "498 labeled road images. 10-channel features: 7 grayscale texture/edge "
            "channels + 3 HSV color channels. Requires model_rf.joblib."
        ),
    }
    st.sidebar.info(method_info[method])

    cfg = build_config_from_sidebar()

    # RF model status
    if method.startswith("Advanced"):
        clf = get_rf_model()
        if clf is None:
            st.error(
                "model_rf.joblib not found. "
                "Train the model first by running Cell 6 in main_training.ipynb, "
                "then restart this app."
            )
            return
    else:
        clf = None

    # Image upload
    st.subheader("Upload Road Image")
    uploaded_img = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_img is None:
        st.info("Upload a road image to start detection.")
        return

    bgr = load_image(uploaded_img)
    if bgr is None:
        st.error("Could not read the uploaded image.")
        return

    if st.button("Detect Potholes", type="primary"):
        with st.spinner("Running detection..."):
            if clf is not None:
                pred_mask, road_mask, road_valid = run_rf(bgr, clf, cfg)
            else:
                pred_mask, road_mask, road_valid = run_baseline(bgr, cfg)

        st.subheader("Detection Results")
        display_results(bgr, road_mask, pred_mask, road_valid)

        st.download_button(
            "Download Predicted Mask (PNG)",
            data=mask_to_bytes(pred_mask),
            file_name="predicted_mask.png",
            mime="image/png"
        )

        # Optional ground truth upload
        st.subheader("Evaluate Against Ground Truth (optional)")
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
                st.error("Could not read the ground truth mask.")


if __name__ == "__main__":
    main()
