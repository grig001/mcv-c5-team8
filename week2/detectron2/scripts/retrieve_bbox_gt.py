import os
import numpy as np
from pycocotools import mask

# PATHS (Set these correctly)
instances_txt_dir = "instances_txt/"  # Folder with RLE mask files
output_bboxes_dir = "w2/gt_bboxes/"  # Folder to save output
output_masks_dir = "w2/gt_masks/"

# Create output directory if it doesn't exist
def rle_to_bbox_and_mask(rle_str, img_height, img_width):
    """Convert RLE mask to bounding box and binary mask"""
    rle = {'size': [img_height, img_width], 'counts': rle_str.encode('utf-8')}
    binary_mask = mask.decode(rle)  # Convert RLE to binary mask

    # Get coordinates of non-zero (object) pixels
    y_indices, x_indices = np.where(binary_mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None  # No object detected

    # Bounding box (x_min, y_min, x_max, y_max)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    bbox = [x_min, y_min, x_max, y_max]

    return bbox, binary_mask

import numpy as np
from pycocotools import mask as mask_utils
import cv2

def process_instance_file(file_path, output_bbox_file, output_mask_dir):
    """Extract bounding boxes and masks from an instance txt file and save them."""
    gt_bboxes = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        parts = line.strip().split(" ")
        frame_id = int(parts[0])  # Frame number
        obj_id = int(parts[1])  # Unique instance ID
        class_id = int(parts[2])  # Class (Car = 2, Pedestrian = 1)
        img_height = int(parts[3])
        img_width = int(parts[4])
        rle_str = " ".join(parts[5:])  # Remaining part is RLE-encoded mask

        bbox, mask = rle_to_bbox_and_mask(rle_str, img_height, img_width)
        if bbox:
            gt_bboxes.append([frame_id, obj_id, class_id, *bbox])

            # Save mask as an image
            mask_filename = f"{output_mask_dir}/frame_{frame_id}_id_{obj_id}.png"
            cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))  # Convert mask to 8-bit

    # Save bounding boxes to separate file for each sequence
    with open(output_bbox_file, "w") as f_out:
        for bbox in gt_bboxes:
            f_out.write(" ".join(map(str, bbox)) + "\n")

    print(f"Extracted {len(gt_bboxes)} ground truth bounding boxes for {file_path}.")
    print(f"Saved masks in {output_mask_dir} and bounding boxes in {output_bbox_file}")

import os

# Process all sequence files
for file_name in sorted(os.listdir(instances_txt_dir)):
    file_path = os.path.join(instances_txt_dir, file_name)
    sequence_id = file_name.split(".")[0]  # Extract sequence number (e.g., "0000" from "0000.txt")

    output_bbox_file = os.path.join(output_bboxes_dir, f"gt_bboxes_{sequence_id}.txt")
    output_mask_dir = os.path.join(output_masks_dir, f"masks_{sequence_id}")  # Separate folder for each sequence

    os.makedirs(output_mask_dir, exist_ok=True)  # Ensure mask directory exists

    process_instance_file(file_path, output_bbox_file, output_mask_dir)

