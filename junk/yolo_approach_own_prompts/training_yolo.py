import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -----------------
# 1. Training Setup
# -----------------
DATASET_YAML = "dataset_yolo.yaml"   # path to your dataset.yaml
EPOCHS = 100
IMGSZ = 640
MODEL = "yolov8n.pt"            # nano model (fast + light)

# -----------------
# 2. Train YOLOv8n
# -----------------
model = YOLO(MODEL)

results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    device=0,            # use GPU 0
    mosaic=1.0,          # strong mosaic augmentation
    copy_paste=0.3,      # patch augmentation
    mixup=0.2,           # blend images
    project="runs",      # save inside runs/detect/
    name="yolov8n_custom"
)

# -----------------
# 3. Generate Plots
# -----------------
results_dir = results.save_dir  # e.g. runs/detect/yolov8n_custom
csv_file = os.path.join(results_dir, "results.csv")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)

    # Loss curves
    plt.figure(figsize=(8,6))
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
    if "train/dfl_loss" in df:  # DFL loss (YOLOv8 regression)
        plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss")

    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss")
    if "val/dfl_loss" in df:
        plt.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLOv8 Training Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "loss_curves.png"), dpi=300)

    # mAP curves
    plt.figure(figsize=(8,6))
    if "metrics/mAP50-95(B)" in df:
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    if "metrics/mAP50(B)" in df:
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("YOLOv8 mAP over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "map_curves.png"), dpi=300)

    print(f"✅ Training finished. Plots saved in {results_dir}")
else:
    print("⚠️ results.csv not found. Did training run correctly?")
