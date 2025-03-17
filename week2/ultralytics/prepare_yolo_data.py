#!/usr/bin/env python3
import os
import shutil
import cv2
import numpy as np
import random
from pycocotools import mask as maskUtils


def create_yolo_substructure(dest_dir, subset="train"):

    images_dir = os.path.join(dest_dir, subset, "images")
    labels_dir = os.path.join(dest_dir, subset, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir


def parse_annotation_line(line):

    parts = line.strip().split(" ")
    if len(parts) < 6:
        return None
    try:
        frame, track_id, class_id, img_height, img_width = map(int, parts[:5])
    except ValueError as e:
        print("Error parsing numeric fields:", e)
        return None
    mask_data_str = " ".join(parts[5:])
    return frame, track_id, class_id, img_height, img_width, mask_data_str


def process_sequence(seq_id, src_img_dir, src_ann_dir, images_dir, labels_dir, class_mapping):
    """Process all frames for a given sequence."""
    seq_folder = os.path.join(src_img_dir, seq_id)
    ann_file = os.path.join(src_ann_dir, f"{seq_id}.txt")
    if not os.path.exists(ann_file):
        print(f"Annotation file for sequence {seq_id} not found at {ann_file}.")
        return

    # Read annotations
    annotations_by_frame = {}
    with open(ann_file, "r") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed:
                frame, track_id, class_id, ann_img_height, ann_img_width, mask_data_str = parsed
                annotations_by_frame.setdefault(frame, []).append(
                    (track_id, class_id, ann_img_height, ann_img_width, mask_data_str)
                )

    # Process each frame
    for frame, ann_list in annotations_by_frame.items():
        filename = f"{frame:06d}.png"
        src_image_path = os.path.join(seq_folder, filename)
        if not os.path.exists(src_image_path):
            print(f"Image {src_image_path} not found.")
            continue

        img = cv2.imread(src_image_path)
        if img is None:
            print(f"Failed to load image {src_image_path}.")
            continue
        height, width = img.shape[:2]

        # Copy image to YOLO dataset
        new_filename = f"{seq_id}_{filename}"
        dest_image_path = os.path.join(images_dir, new_filename)
        shutil.copy(src_image_path, dest_image_path)

        # Process annotations
        yolo_lines = []
        for track_id, class_id, ann_img_height, ann_img_width, mask_data_str in ann_list:
            rle = {"counts": mask_data_str, "size": [ann_img_height, ann_img_width]}
            try:
                binary_mask = maskUtils.decode(rle)
            except Exception as e:
                print(f"Error decoding mask for image {src_image_path}: {e}")
                continue

            # Resize mask if needed
            if binary_mask.shape[0] != height or binary_mask.shape[1] != width:
                binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Extract segmentation polygon
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            polygon = max(contours, key=cv2.contourArea).flatten()

            # Normalize polygon coordinates
            normalized_polygon = [polygon[i] / width if i % 2 == 0 else polygon[i] / height for i in range(len(polygon))]
            segmentation_str = " ".join(f"{p:.6f}" for p in normalized_polygon)

            # Map class_id to YOLO class index
            if class_id in class_mapping:
                yolo_class = class_mapping[class_id]
            else:
                continue  # Skip if class is not mapped

            # Format: <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
            yolo_lines.append(f"{yolo_class} {segmentation_str}")

        # Write YOLO label file
        label_filename = new_filename.replace(".png", ".txt")
        dest_label_path = os.path.join(labels_dir, label_filename)
        with open(dest_label_path, "w") as lf:
            lf.write("\n".join(yolo_lines))


def split_train_val(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.15):

    all_images = sorted(os.listdir(train_images_dir))
    random.shuffle(all_images)

    num_val = int(len(all_images) * split_ratio)
    val_images = all_images[:num_val]
    train_images = all_images[num_val:]

    print(f"Splitting: {len(train_images)} train images, {len(val_images)} val images.")

    for img_file in val_images:
        src_img_path = os.path.join(train_images_dir, img_file)
        src_lbl_path = os.path.join(train_labels_dir, img_file.replace(".png", ".txt"))

        dest_img_path = os.path.join(val_images_dir, img_file)
        dest_lbl_path = os.path.join(val_labels_dir, img_file.replace(".png", ".txt"))

        shutil.move(src_img_path, dest_img_path)
        if os.path.exists(src_lbl_path):
            shutil.move(src_lbl_path, dest_lbl_path)


def main(mode):

    src_base = "/ghome/c5mcv08/team8_split_KITTI-MOTS/"
    dest_dir = "/ghome/c5mcv08/team8_split_KITTI-MOTS_YOLO"

    # Create YOLO directory structures
    train_images_dir, train_labels_dir = create_yolo_substructure(dest_dir, "train")
    val_images_dir, val_labels_dir = create_yolo_substructure(dest_dir, "val")
    eval_images_dir, eval_labels_dir = create_yolo_substructure(dest_dir, "eval")

    # Define class mapping
    class_mapping = {
        "pretrain": {1: 1, 2: 0},
        "finetune": {1: 1, 2: 0},
    }

    # Dataset paths
    src_train_dir = os.path.join(src_base, "train")
    src_eval_dir = os.path.join(src_base, "eval")
    train_ann_dir = os.path.join(src_base, "instances_txt", "train")
    eval_ann_dir = os.path.join(src_base, "instances_txt", "eval")

    # Process training sequences
    for seq_id in os.listdir(src_train_dir):
        if os.path.isdir(os.path.join(src_train_dir, seq_id)):
            print(f"Processing training sequence: {seq_id}")
            process_sequence(seq_id, src_train_dir, train_ann_dir, train_images_dir, train_labels_dir, class_mapping[mode])

    # Perform train-val split
    split_train_val(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.15)

    # Process evaluation sequences
    for seq_id in os.listdir(src_eval_dir):
        if os.path.isdir(os.path.join(src_eval_dir, seq_id)):
            print(f"Processing evaluation sequence: {seq_id}")
            process_sequence(seq_id, src_eval_dir, eval_ann_dir, eval_images_dir, eval_labels_dir, class_mapping[mode])


if __name__ == "__main__":
    main(mode="pretrain")
