from ultralytics import YOLO

# Load yolov8n-seg model
model = YOLO("models/yolov8n-seg.pt")

# Train with augmentations
model.train(
    data="datasets/yolo_fine_tune.yaml",
    epochs=10,
    batch=16,
    imgsz=640,
    device="cuda",
    augment=True,
    degrees=10,
    scale=0.2,
    flipud=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.2,
    hsv_v=0.2,
    mosaic=0.2
)

print("Training with augmentations completed!")
