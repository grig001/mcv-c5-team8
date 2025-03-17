from ultralytics import YOLO

# Define the YOLO dataset configuration file
DATASET_YAML = "datasets/yolo_fine_tune.yaml"

# Load the fine tunded yolov8n-seg model
model = YOLO("runs/segment/train4/weights/best.pt")

# Run YOLO evaluation on the test dataset
metrics = model.val(data=DATASET_YAML, save_json=True, split="test", workers=4)

print("Evaluation completed!", metrics)
