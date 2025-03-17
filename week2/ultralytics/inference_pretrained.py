from ultralytics import YOLO

# Define the YOLO dataset configuration file
DATASET_YAML = "datasets/yolo_pretrained.yaml"

# Load the YOLOv8 segmentation model
model = YOLO("models/yolov8n-seg.pt")

# Run YOLO evaluation on the test dataset
results = model.val(data=DATASET_YAML, save_json=True, split="test", workers=4)

print("Evaluation completed!")
