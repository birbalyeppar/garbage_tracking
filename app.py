import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from datetime import datetime
import os

# Load trained model
model = YOLO('best.pt')

st.title("🚯 Garbage Detection - Garbage Throwing & Garbage Bags")

# Slider for detection confidence threshold
conf_threshold = st.slider(
    "Detection Confidence Threshold",
    0.0, 1.0, 0.4, 0.05,
    key="conf_slider"
)

# Slider for frame skip (to speed up)
frame_skip = st.slider(
    "Process every Nth frame (higher = faster, less accurate)",
    1, 10, 3, 1,
    key="frame_skip"
)

video_file = st.file_uploader(
    "Upload a short video",
    type=["mp4", "mov", "avi"],
    key="video_upload"
)

if video_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Prepare output video path
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if frame_count % frame_skip == 0:
            results = model(frame, conf=conf_threshold)[0]
            annotated_frame = frame.copy()
            alert_triggered = False

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label in ["garbage_throw", "garbage_bag"]:
                    alert_triggered = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Different colors for different labels
                    color = (0, 0, 255) if label == "garbage_throw" else (0, 255, 0)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_frame, f"{label} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )

            out.write(annotated_frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            if alert_triggered:
                st.warning("🚨 Garbage detected!")
        else:
            # If skipping frame, just write the unprocessed frame
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()

    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 Download Annotated Video",
            data=f,
            file_name="garbage_detection_output.mp4",
            mime="video/mp4"
        )
