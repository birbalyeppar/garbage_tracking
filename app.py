import os
import time
import cv2
import torch
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
 
st.set_page_config(page_title="Garbage Detection (Realtime + Download)", layout="wide")
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        if "nvidia" in name:
            return "cuda"
        else:
            print(f"⚠️ Ignoring non-NVIDIA GPU: {name}")
            return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu" 
DEVICE = get_device()
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high") 
except Exception:
    pass
st.write(f"Inference device: **{DEVICE.upper()}**")

@st.cache_resource
def load_model():
    weights_path = hf_hub_download(
        repo_id="birbalk99/garbage-model",
        filename="best.pt"
    )
    model = YOLO(weights_path, task="detect")
    model.to(DEVICE)

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
 
st.title("Realtime Garbage Detection")
st.subheader("Detect and track garbage in video")

video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
video_placeholder = st.empty()
progress = st.empty()
status = st.empty()
stop = st.checkbox("Stop after current frame", value=False)
 
CONF_TH = 0.05
IOU_TH  = 0.50
IMG_SZ  = 640
TARGET_MIN_FPS = 15.0
 
if video_file:
    src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    src_tmp.write(video_file.read())
    src_tmp.flush()
 
    cap = cv2.VideoCapture(src_tmp.name)
    if not cap.isOpened():
        st.error("⚠️ Failed to open the video.")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = TARGET_MIN_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = 1.0 / float(fps)
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
                    break 
                t0 = time.monotonic()
                results = model.predict(
                    frame_bgr,
                    device=DEVICE,
                    half=(DEVICE == "cuda"),
                    imgsz=IMG_SZ,
                    conf=CONF_TH,
                    iou=IOU_TH,
                    verbose=False,
                )
                annotated_bgr = results[0].plot()
                writer.write(annotated_bgr)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB")
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
    with open(out_path, "rb") as f:
        st.download_button(
            label="⬇ Download Processed Video",
            data=f,
            file_name="processed_output.mp4",
            mime="video/mp4",
        )
    try:
        os.unlink(src_tmp.name)
    except Exception:
        pass