import os
import time
import cv2
import torch
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
 
# ---------------------------
# Page & perf setup
# ---------------------------
st.set_page_config(page_title="Garbage Detection (Realtime + Download)", layout="wide")
 
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # MPS available on Apple Silicon (macOS 12.3+ and torch built with mps)
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"
 
DEVICE = get_device()
 
# cuDNN benchmark only helps on CUDA
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
 
# Optional: speed up float32 matmul on newer PyTorch builds
try:
    torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
except Exception:
    pass
 
st.write(f"Inference device: **{DEVICE.upper()}**")
 
# ---------------------------
# Load model (cached, single load for all sessions)
# ---------------------------
@st.cache_resource
def load_model(weights_path: str = "model_train.pt"):
    model = YOLO(weights_path, task="detect")
    model.to(DEVICE)
    # warmup with dummy image (FP16 only on CUDA)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(
        dummy,
        device=DEVICE,
        imgsz=640,
        half=(DEVICE == "cuda"),
        verbose=False,
    )
    return model
 
model = load_model()
CLASS_NAMES = model.names if hasattr(model, "names") else {}
 
# ---------------------------
# UI
# ---------------------------
st.title("Realtime Garbage Detection")
st.subheader("Detect and track garbage in video")

video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
video_placeholder = st.empty()
progress = st.empty()
status = st.empty()
 
# (Optional) Stop early
stop = st.checkbox("Stop after current frame", value=False)
 
# ---------------------------
# Constants (tweak if needed)
# ---------------------------
CONF_TH = 0.05   # lower conf to catch smaller objects
IOU_TH  = 0.50   # slightly relaxed NMS
IMG_SZ  = 640    # solid trade-off for latency/accuracy
TARGET_MIN_FPS = 15.0  # if video fps is too low/unknown, use this
 
if video_file:
    # Save upload to temp file for OpenCV
    src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    src_tmp.write(video_file.read())
    src_tmp.flush()
 
    cap = cv2.VideoCapture(src_tmp.name)
    if not cap.isOpened():
        st.error("⚠️ Failed to open the video.")
        st.stop()
 
    # Read meta
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = TARGET_MIN_FPS  # fallback
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = 1.0 / float(fps)
 
    # Prepare output writer (processed video)
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
 
    processed = 0
    t_last_push = 0.0
 
    status.info("🎬 Processing… (video will play at its native FPS)")
    start_all = time.monotonic()
 
    try:
        with torch.inference_mode():
            while True:
                if stop:
                    break
 
                ok, frame_bgr = cap.read()
                if not ok:
                    break  # 🔚 video finished → exit loop cleanly
 
                t0 = time.monotonic()
 
                # Inference
                results = model.predict(
                    frame_bgr,
                    device=DEVICE,
                    half=(DEVICE == "cuda"),
                    imgsz=IMG_SZ,
                    conf=CONF_TH,
                    iou=IOU_TH,
                    verbose=False,
                )
 
                # Fast annotation (Ultralytics)
                annotated_bgr = results[0].plot()
 
                # Write to output video
                writer.write(annotated_bgr)
 
                # Display throttled to source FPS (no speedup)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB")
 
                # Maintain ~video FPS (avoid faster-than-source playback)
                elapsed = time.monotonic() - t0
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
 
                processed += 1
                if frame_count > 0:
                    progress.progress(min(processed / frame_count, 1.0), text=f"{processed}/{frame_count} frames")
    finally:
        cap.release()
        writer.release()
 
    total_time = time.monotonic() - start_all
    status.success(f"✅ Done! Processed {processed} frames in {total_time:.1f}s")
 
    # Download button for processed video
    with open(out_path, "rb") as f:
        st.download_button(
            label="⬇ Download Processed Video",
            data=f,
            file_name="processed_output.mp4",
            mime="video/mp4",
        )
 
    # Clean up temp source file (keep output until page refresh)
    try:
        os.unlink(src_tmp.name)
    except Exception:
        pass