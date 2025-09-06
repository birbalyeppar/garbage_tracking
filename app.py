import os
import time
import cv2
import torch
import uuid
import pandas as pd
from datetime import datetime
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Garbage Detection (Realtime + Download)", layout="wide")
active_clips = {}

def ensure_csv_exists(csv_path: str):
    """Ensure the CSV file exists with proper headers."""
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["clip_id", "class_detected", "start_time", "end_time", "video_path", "location"])
        df.to_csv(csv_path, index=False)


def save_clip(clip_id: str, frames: list, fps: int, clip_output_dir: str) -> str:
    """Save frames into a video file and return the file path."""
    os.makedirs(clip_output_dir, exist_ok=True)
    clip_filename = f"{clip_id}.mp4"
    clip_path = os.path.join(clip_output_dir, clip_filename)

    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()

    return clip_path


def log_clip_to_csv(csv_path: str, clip_id: str, key: str, start_time: float, end_time: float, clip_path: str, location: str):
    """Log a saved clip to CSV."""
    start_dt = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_dt = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(csv_path)
    df.loc[len(df)] = [clip_id, key, start_dt, end_dt, clip_path, location]
    df.to_csv(csv_path, index=False)

def process_detection_clips(
    frames_with_time,
    results_list,
    selected_classes,
    class_names,
    buffer_sec=5,
    after_sec=10,
    max_gap_sec=3,
    clip_output_dir="clips",
    csv_path="detections_log.csv",
    default_location="Default_Location_Name",
    fps=15,
    show_labels=True
):
    global active_clips
    ensure_csv_exists(csv_path)
    for (t_now, frame), results in zip(frames_with_time, results_list):
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = class_names[cls_id]
            track_id = int(box.id[0].item()) if box.id is not None else None
            print(f"Detected:---------------------------------- class={cls_name}, track_id={track_id}, time={t_now}")

            if cls_name in selected_classes and track_id is not None:
                key = f"{cls_name}_{track_id}"

                if key not in active_clips:
                    active_clips[key] = {
                        "frames": [],
                        "start_time": t_now - buffer_sec,
                        "last_seen": t_now,
                    }

                # 🔑 Sirf is box ke liye frame save karo
                frame_to_save = frame.copy()
                if show_labels:
                    # Box ke coordinates leke label draw karo
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame_to_save, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                    cv2.putText(frame_to_save, cls_name, (xyxy[0], xyxy[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                active_clips[key]["frames"].append(frame_to_save)
                active_clips[key]["last_seen"] = t_now
        # Finalize inactive clips (object gone from scene)
        to_finalize = [
            key for key, clip in active_clips.items()
            if (t_now - clip["last_seen"]) > max_gap_sec
        ]

        for key in to_finalize:
            clip = active_clips[key]
            clip_id = str(uuid.uuid4())

            # Save video
            clip_path = save_clip(clip_id, clip["frames"], fps, clip_output_dir)

            # Log entry
            log_clip_to_csv(
                csv_path, clip_id, key,
                clip["start_time"], clip["last_seen"] + after_sec,
                clip_path, default_location
            )

            del active_clips[key]



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

st.sidebar.header("Select Classes to Detect")
show_labels = st.sidebar.checkbox("Show labels on clips", value=True)
selected_classes = st.sidebar.multiselect(
    "Choose Detection Classes",
    options=list(CLASS_NAMES.values()),  # List of model class names
    default=list(CLASS_NAMES.values())   # By default, all selected
)

selected_class_ids = [k for k,v in CLASS_NAMES.items() if v in selected_classes]
 
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
        frames_with_time = []
        results_list = []
        with torch.inference_mode():
            while True:
                if stop:
                    break
 
                ok, frame_bgr = cap.read()
                if not ok:
                    break 
                t0 = time.monotonic()
                t_now = time.time()
                results = model.track(
                    frame_bgr,
                    device=DEVICE,
                    half=(DEVICE == "cuda"),
                    imgsz=IMG_SZ,
                    conf=CONF_TH,
                    iou=IOU_TH,
                    classes=selected_class_ids,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )
                annotated_bgr = results[0].plot()
                writer.write(annotated_bgr)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB")
                frames_with_time.append((t_now, frame_bgr.copy()))
                results_list.append(results)
                process_detection_clips(
                    frames_with_time,
                    results_list,
                    selected_classes,
                    CLASS_NAMES,
                    fps=fps
                )
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