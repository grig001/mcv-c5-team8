import os

# This script is supposed to convert the gt files to required yolo format.

base_dir = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C5_project/KITTI-MOTS/yolo_format/dataset"
image_root = os.path.join(base_dir, "images")
label_root = os.path.join(base_dir, "labels")


class_map = {"Pedestrian": 0, "Car": 1}

#conversion of gt bbox to normalized yolo format
def convert_bbox(img_width, img_height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / (2 * img_width)
    y_center = (ymin + ymax) / (2 * img_height)
    box_width = (xmax - xmin) / img_width
    box_height = (ymax - ymin) / img_height
    return f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

#processing train and val gt files
for split in ["train", "val"]:
    split_image_path = os.path.join(image_root, split)  
    split_label_path = os.path.join(label_root, split)  

    for gt_file in os.listdir(split_label_path):  
        if gt_file.startswith("gt_bboxes_") and gt_file.endswith(".txt"):
            seq_id = gt_file.replace("gt_bboxes_", "").replace(".txt", "")  

            seq_label_dir = os.path.join(split_label_path, seq_id)  
            os.makedirs(seq_label_dir, exist_ok=True)  

            with open(os.path.join(split_label_path, gt_file), "r") as f:
                annotations = {}

                for line in f:
                    parts = line.strip().split()
                    img_name = parts[0]  
                    label = parts[1]  
                    xmin, ymin, xmax, ymax = map(int, parts[3:7])

                    img_width, img_height = 1920, 1080
                    bbox = convert_bbox(img_width, img_height, xmin, ymin, xmax, ymax)

                    img_txt_name = img_name.replace(".png", ".txt")
                    if img_txt_name not in annotations:
                        annotations[img_txt_name] = []
                    annotations[img_txt_name].append(f"{class_map[label]} {bbox}")

                for img_txt_name, label_data in annotations.items():
                    label_file_path = os.path.join(seq_label_dir, img_txt_name)
                    with open(label_file_path, "w") as f_out:
                        f_out.write("\n".join(label_data))

print("Annotations successfully converted to YOLO format!")

