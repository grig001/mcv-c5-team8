import os
import torch
from ultralytics import YOLO

# Define dataset path
base_dir = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C5_project/KITTI-MOTS/yolo_format/dataset"
dataset_yaml = os.path.join(base_dir, "dataset.yaml")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model (Choose the appropriate size: n, s, m, l, x)
model = YOLO("yolov8s.pt")  # Change to 'yolov8n.pt' for faster training

n_epoch = 10
batch = 16
imagesize = 640
workers = 4
augment1 = 0.0
augment2 = 0.5
augment3 = 1.0

# Train the model
model.train(
    data=dataset_yaml,
    epochs=n_epoch,  # Adjust as needed
    batch=batch,
    imgsz=imagesize,
    workers=4,
    device=device,
    flipud = augment1,
    fliplr = augment2,
    mosaic = augment3
)

# Save the trained model
trained_model_path = os.path.join(base_dir, "yolov8_kitti_trained.pt")
model.export(format="torchscript")  # Export the model

print(f"Training complete! Model saved as {trained_model_path}")

print(f"\nModel was trained with following parameters:\n epochs = {n_epoch}\nbatchsize = {batch}\nimagesize = {imagesize}\nworkers = {workers}\nflpupd = {augment1}\nfliplr = {augment2}\nmosaic = {augment3}" )

metrics = model.val(data="dataset.yaml", batch=16, imgsz=640, device="cuda")
print(metrics)
