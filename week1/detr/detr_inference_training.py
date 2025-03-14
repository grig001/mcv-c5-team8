import os
import glob
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

MODEL_NAME = "facebook/detr-resnet-50"
KITTI_MOTS_TRAINING_DIR = "/home/c5mcv08/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_DIR = "output_detr_training_gt_format"

DETR_TO_KITTI_CLASSES = {1: "Pedestrian", 3: "Car"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model = DetrForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sequence_dirs = sorted(glob.glob(os.path.join(KITTI_MOTS_TRAINING_DIR, "*")))

for seq_dir in sequence_dirs:
    seq_name = os.path.basename(seq_dir)
    print(f"Processing Sequence: {seq_name}")

    image_paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))

    output_file_path = os.path.join(OUTPUT_DIR, f"predictions_{seq_name}.txt")
    with open(output_file_path, "w") as f_out:
        for img_path in image_paths:
            image_name = os.path.basename(img_path)
            print(f"  📷 Processing Image: {image_name}")

            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size  # Get actual image size
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.softmax(-1)[0, :, :-1]
            boxes = outputs.pred_boxes[0].cpu().numpy()

            max_scores, labels = logits.max(-1)
            keep = max_scores > 0.5

            for i in range(len(keep)):
                if keep[i]:
                    class_id = labels[i].item()

                    if class_id not in DETR_TO_KITTI_CLASSES:
                        continue

                    class_name = DETR_TO_KITTI_CLASSES[class_id]

                    score = max_scores[i].item()
                    center_x, center_y, width, height = boxes[i]

                    x_min = int((center_x - width / 2) * img_w)
                    y_min = int((center_y - height / 2) * img_h)
                    x_max = int((center_x + width / 2) * img_w)
                    y_max = int((center_y + height / 2) * img_h)

                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(img_w, x_max), min(img_h, y_max)

                    f_out.write(f"{image_name} {class_name} {score:.2f} {x_min} {y_min} {x_max} {y_max}\n")

print("Predictions saved in:", OUTPUT_DIR)
