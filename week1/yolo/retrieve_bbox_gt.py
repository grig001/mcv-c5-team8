import os
import numpy as np
from pycocotools import mask

# This script reads original groundtruth txt files and converts them to yolo like gt txt files

instances_txt_dir = "../KITTI-MOTS/instances_txt"  
output_bboxes_dir = "gt_bboxes"  

CLASS_MAPPING = {2: "Pedestrian", 1: "Car"}

os.makedirs(output_bboxes_dir, exist_ok=True)

def rle_to_bbox(rle_str, img_height, img_width):
    """Convert RLE mask to bounding box [x_min, y_min, x_max, y_max]"""
    rle = {'size': [img_height, img_width], 'counts': rle_str.encode('utf-8')}
    binary_mask = mask.decode(rle) 

    y_indices, x_indices = np.where(binary_mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    return [x_min, y_min, x_max, y_max]

def process_instance_file(file_path, output_file, sequence_id):
    """Extract bounding boxes from a single instance txt file and save in YOLO format."""
    gt_bboxes = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split(" ")

        if len(parts) < 6:  
            continue

        frame_id = int(parts[0])  
        obj_id = int(parts[1])  
        class_id = int(parts[2])  

        class_name = CLASS_MAPPING.get(class_id, "Unknown")

        if class_name == "Unknown":
            continue  

        img_height = int(parts[3])
        img_width = int(parts[4])
        rle_str = " ".join(parts[5:])  
        
        bbox = rle_to_bbox(rle_str, img_height, img_width)
        if bbox:
            img_name = f"{frame_id:06d}.png"
            gt_bboxes.append(f"{img_name} {class_name} 1.00 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    with open(output_file, "w") as f_out:
        for line in gt_bboxes:
            f_out.write(line + "\n")

    print(f"Extracted {len(gt_bboxes)} ground truth bounding boxes for {file_path}. Saved in {output_file}")

def clean_txt_files(output_dir):
    """Remove lines containing 'Unknown' from all saved txt files."""
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "r") as f:
            lines = f.readlines()

        cleaned_lines = [line for line in lines if "Unknown" not in line]

        with open(file_path, "w") as f_out:
            f_out.writelines(cleaned_lines)

        print(f"Cleaned {file_path}, removed {len(lines) - len(cleaned_lines)} unwanted lines.")

for file_name in sorted(os.listdir(instances_txt_dir)):
    file_path = os.path.join(instances_txt_dir, file_name)
    sequence_id = file_name.split(".")[0]
    output_file = os.path.join(output_bboxes_dir, f"gt_bboxes_{sequence_id}.txt")

    process_instance_file(file_path, output_file, sequence_id)

# **Final cleanup step**
clean_txt_files(output_bboxes_dir)
