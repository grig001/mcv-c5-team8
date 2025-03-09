import os
import numpy as np
from pycocotools import mask

# PATHS (Set these correctly)
instances_txt_dir = "../KITTI-MOTS/instances_txt"  # Folder with RLE mask files
output_bboxes_dir = "gt_bboxes"  # Folder to save GT bbox files

# Corrected Class Mapping (KITTI MOTS → YOLO)
CLASS_MAPPING = {2: "Pedestrian", 1: "Car"}  # Corrected mapping

# Create output directory if it doesn't exist
os.makedirs(output_bboxes_dir, exist_ok=True)

def rle_to_bbox(rle_str, img_height, img_width):
    """Convert RLE mask to bounding box [x_min, y_min, x_max, y_max]"""
    rle = {'size': [img_height, img_width], 'counts': rle_str.encode('utf-8')}
    binary_mask = mask.decode(rle)  # Convert RLE to binary mask

    # Get coordinates of non-zero (object) pixels
    y_indices, x_indices = np.where(binary_mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No object detected

    # Bounding box (x_min, y_min, x_max, y_max)
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

        if len(parts) < 6:  # Ensure valid line format
            continue

        frame_id = int(parts[0])  # Frame number
        obj_id = int(parts[1])  # Unique instance ID
        class_id = int(parts[2])  # Class ID

        # Convert class ID to class name
        class_name = CLASS_MAPPING.get(class_id, "Unknown")

        if class_name == "Unknown":
            continue  # Skip "Unknown" class (including 10000)

        img_height = int(parts[3])
        img_width = int(parts[4])
        rle_str = " ".join(parts[5:])  # Remaining part is RLE-encoded mask
        
        bbox = rle_to_bbox(rle_str, img_height, img_width)
        if bbox:
            # Format: "000000.png Pedestrian 1.00 x_min y_min x_max y_max"
            img_name = f"{frame_id:06d}.png"
            gt_bboxes.append(f"{img_name} {class_name} 1.00 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    # Save to output file
    with open(output_file, "w") as f_out:
        for line in gt_bboxes:
            f_out.write(line + "\n")

    print(f"Extracted {len(gt_bboxes)} ground truth bounding boxes for {file_path}. Saved in {output_file}")

def clean_txt_files(output_dir):
    """Remove lines containing 'Unknown' from all saved txt files."""
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)

        # Read and filter lines
        with open(file_path, "r") as f:
            lines = f.readlines()

        cleaned_lines = [line for line in lines if "Unknown" not in line]

        # Overwrite file with cleaned lines
        with open(file_path, "w") as f_out:
            f_out.writelines(cleaned_lines)

        print(f"Cleaned {file_path}, removed {len(lines) - len(cleaned_lines)} unwanted lines.")

# Process all sequence files
for file_name in sorted(os.listdir(instances_txt_dir)):
    file_path = os.path.join(instances_txt_dir, file_name)
    sequence_id = file_name.split(".")[0]  # Extract sequence number (e.g., "0000" from "0000.txt")
    output_file = os.path.join(output_bboxes_dir, f"gt_bboxes_{sequence_id}.txt")

    process_instance_file(file_path, output_file, sequence_id)

# **Final cleanup step**
clean_txt_files(output_bboxes_dir)
