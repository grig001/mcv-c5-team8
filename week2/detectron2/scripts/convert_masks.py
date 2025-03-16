import json
import os
import cv2
import numpy as np
from pycocotools import mask

image_path = "dataset/image_02/"
bboxes_input_folder = "w2/gt_bboxes/"
masks_input_folder = "w2/gt_masks/"

output_folder = "w2/json_annotations/"

os.makedirs(output_folder, exist_ok=True)


# Predefined category mapping
category_map = {1: 1, 2: 2}  # class_1 → id: 1, class_2 → id: 2
categories = [
    {"id": 1, "name": "class_1", "supercategory": "object"},
    {"id": 2, "name": "class_2", "supercategory": "object"}
]


for file_idx in range(21):
    gt_file = os.path.join(bboxes_input_folder, f"gt_bboxes_{file_idx:04d}.txt")
    output_json = os.path.join(output_folder, f"gt_bboxes_{file_idx:04d}.json")
    mask_folder = os.path.join(masks_input_folder, f"masks_{file_idx:04d}")  # Path to masks

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    # category_map = {}  
    image_id_map = []  
    annotation_id = 1  

    if not os.path.exists(gt_file):
        print(f"File {gt_file} not found, skipping...")
        continue

    with open(gt_file, "r") as f:
        lines = f.readlines()


    for line in lines:
        parts = list(map(int, line.strip().split()))
        frame_id, obj_id, class_id, x_min, y_min, x_max, y_max = parts

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        if class_id == 10:
            class_id = 1
    
        if frame_id not in image_id_map:
            image_id_map.append(frame_id)

            folder_index = str(file_idx).zfill(4)
            img_path = os.path.join(image_path, folder_index, f"{frame_id:06d}.png")
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Image {img_path} not found, skipping...")
                continue

            height, width, _ = img.shape

            coco_data["images"].append({
                "id": int(1e5 * file_idx + frame_id),
                "file_name": f"{frame_id:06d}.png",
                "height": height,
                "width": width
            })


        # Load and encode mask
        mask_path = os.path.join(mask_folder, f"frame_{frame_id}_id_{obj_id}.png")
        if os.path.exists(mask_path):
            im_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            im_mask = np.where(im_mask > 0, 1, 0).astype(np.uint8)  # Convert to binary mask
            rle = mask.encode(np.asfortranarray(im_mask))  # Convert to RLE format
            rle["counts"] = rle["counts"].decode("utf-8")  # Ensure it's JSON serializable
        else:
            print(f"Warning: Mask {mask_path} not found, using empty mask.")
            rle = {"size": [height, width], "counts": ""}  # Empty mask

        coco_data["annotations"].append({
            "id": int(1e5 * file_idx + annotation_id),
            "image_id": int(1e5 * file_idx + frame_id),
            "category_id": category_map[class_id],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "segmentation": rle,
            "iscrowd": 0
        })
        annotation_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Saved {output_json}")
