import os

# Base dataset path
base_dir = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C5_project/KITTI-MOTS/yolo_format/dataset"

# Paths for images and labels
image_root = os.path.join(base_dir, "images")
label_root = os.path.join(base_dir, "labels")

# Class mapping (modify if you have more classes)
class_map = {"Pedestrian": 0, "Car": 1}

# Function to convert bbox to YOLO format
def convert_bbox(img_width, img_height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / (2 * img_width)
    y_center = (ymin + ymax) / (2 * img_height)
    box_width = (xmax - xmin) / img_width
    box_height = (ymax - ymin) / img_height
    return f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

# Process train and val sets
for split in ["train", "val"]:
    split_image_path = os.path.join(image_root, split)  # e.g., images/train/
    split_label_path = os.path.join(label_root, split)  # e.g., labels/train/

    for gt_file in os.listdir(split_label_path):  # Loop through gt_bboxes_xxxx.txt files
        if gt_file.startswith("gt_bboxes_") and gt_file.endswith(".txt"):
            seq_id = gt_file.replace("gt_bboxes_", "").replace(".txt", "")  # Extract sequence number (e.g., "0000")

            seq_label_dir = os.path.join(split_label_path, seq_id)  # labels/train/0000/
            os.makedirs(seq_label_dir, exist_ok=True)  # Ensure sequence label directory exists

            with open(os.path.join(split_label_path, gt_file), "r") as f:
                annotations = {}

                for line in f:
                    parts = line.strip().split()
                    img_name = parts[0]  # e.g., "000000.png"
                    label = parts[1]  # e.g., "Pedestrian"
                    xmin, ymin, xmax, ymax = map(int, parts[3:7])

                    # Assuming all images are 1920x1080 (update if needed)
                    img_width, img_height = 1242, 375
                    bbox = convert_bbox(img_width, img_height, xmin, ymin, xmax, ymax)

                    # Store annotation per image
                    img_txt_name = img_name.replace(".png", ".txt")
                    if img_txt_name not in annotations:
                        annotations[img_txt_name] = []
                    annotations[img_txt_name].append(f"{class_map[label]} {bbox}")

                # Save individual label files
                for img_txt_name, label_data in annotations.items():
                    label_file_path = os.path.join(seq_label_dir, img_txt_name)
                    with open(label_file_path, "w") as f_out:
                        f_out.write("\n".join(label_data))

print("Annotations successfully converted to YOLO format!")

