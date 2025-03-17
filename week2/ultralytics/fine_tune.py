from ultralytics import YOLO

# Load model
model = YOLO("models/yolov8n-seg.pt")

# Trains model
model.train(
    data="datasets/yolo_fine_tune.yaml",
    epochs=10,
    batch=16,
    imgsz=640,
    device="cuda"
)

print("Training completed!")
