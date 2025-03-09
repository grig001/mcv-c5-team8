import os
import glob
import torch
from ultralytics import YOLO

# This script is supposed to compute detections from YOLO

MODEL_PATH = "./models/yolov8n.pt"
KITTI_MOTS_TEST_DIR = "../KITTI-MOTS/training/image_02"
OUTPUT_DIR = "output_yolov8_kitti_mots_txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running YOLOv8 on: {device.upper()}")

model = YOLO(MODEL_PATH).to(device)
os.makedirs(OUTPUT_DIR, exist_ok=True)

KITTI_MOTS_CLASSES = {"car": "Car", "person": "Pedestrian"}

sequence_dirs = sorted(glob.glob(os.path.join(KITTI_MOTS_TEST_DIR, "*")))

if len(sequence_dirs) == 0:
    print("No test sequences found! Check dataset path.")
    exit()

print(f"Found {len(sequence_dirs)} test sequences.")

for seq_dir in sequence_dirs:
    seq_name = os.path.basename(seq_dir)
    print(f"Processing Sequence: {seq_name}")

    image_paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))

    if len(image_paths) == 0:
        print(f"No images found in {seq_name}. Skipping...")
        continue

    output_file_path = os.path.join(OUTPUT_DIR, f"predictions_{seq_name}.txt")
    with open(output_file_path, "w") as f_out:
        for img_path in image_paths:
            image_name = os.path.basename(img_path)
            print(f"Image: {image_name}")

            results = model(img_path, device=DEVICE)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  
                confs = result.boxes.conf.cpu().numpy()  
                classes = result.boxes.cls.cpu().numpy()  

                for i in range(len(classes)):
                    class_name = model.names[int(classes[i])].lower()

                    if class_name in KITTI_MOTS_CLASSES:
                        label = KITTI_MOTS_CLASSES[class_name]
                        score = confs[i]
                        x_min, y_min, x_max, y_max = boxes[i]

                        f_out.write(f"{image_name} {label} {score:.2f} {int(x_min)} {int(y_min)} {int(x_max)} {int(y_max)}\n")

print("YOLOv8 Inference completed! Predictions saved in:", OUTPUT_DIR)
