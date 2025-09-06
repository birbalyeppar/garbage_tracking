# utils.py
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    weights_path = hf_hub_download(repo_id="birbalk99/garbage-model", filename="best.pt")
    model = YOLO(weights_path, task="detect")
    model.to(DEVICE)
    dummy = np.zeros((640,640,3), dtype=np.uint8)
    model.predict(dummy, device=DEVICE, imgsz=640, half=(DEVICE=="cuda"), verbose=False)
    return model
