import os
import torch
from ultralytics import YOLO

# This script trains a yolo model from ultralytics library using the KITTI-MOTS dataset

#dataset pahts
base_dir = "../yolo_format/dataset"
dataset_yaml = os.path.join(base_dir, "dataset.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#choose good size of model (n, s, m, l, xl)
model = YOLO("yolov8s.pt")

#training parameters
n_epoch = 30
batch = 32
imagesize = 640
workers = 4
augment1 = 0.0
augment2 = 0.5
augment3 = 0.0

#training
model.train(
    data=dataset_yaml,
    epochs=n_epoch,
    batch=batch,
    imgsz=imagesize,
    workers=4,
    device=device,
    flipud = augment1,
    fliplr = augment2,
    mosaic = augment3
)

trained_model_path = os.path.join(base_dir, f"yolov8_kitti_trained_{n_epoch}_{batch}_{augment1}_{augment2}_{augment3}.pt")
model.export(format="torchscript")  

print(f"Training complete! Model saved as {trained_model_path}")

print(f"\nModel was trained with following parameters:\nepochs = {n_epoch}\nbatchsize = {batch}\nimagesize = {imagesize}\nworkers = {workers}\nflpupd = {augment1}\nfliplr = {augment2}\nmosaic = {augment3}" )

#evaluate trained model
metrics = model.val(data=dataset_yaml, batch=32, imgsz=640, device="cuda")
print(metrics)
