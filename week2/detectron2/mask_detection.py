import os
import json
import torch
import numpy as np
import cv2
import time

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools import mask as mask_utils

# List of validation sequences
val = [2, 6, 7, 8, 10, 13, 14, 16, 18]
# COCO_CATEGORY_MAPPING = {0: 2, 2: 1}

def save_json_safe(data, filename):
    """ Converts NumPy types to standard Python types before saving JSON """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # Assuming 2 classes
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Pre-trained weights
    # cfg.MODEL.WEIGHTS = "output_masks/model_1/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda:0"
    return cfg

def run_inference(cfg, image_path):
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    outputs = predictor(image)
    return outputs

def encode_mask(mask):
    """Convert binary mask to RLE format."""
    mask = np.asfortranarray(mask)  # Ensure it's in Fortran order
    rle = mask_utils.encode(mask)  # RLE encoding
    rle["counts"] = rle["counts"].decode("utf-8")  # Convert from bytes to string
    return rle

def format_detections(outputs, image_id):
    detections = []
    
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_masks = outputs["instances"].pred_masks.cpu().numpy()  # Extract predicted masks
    
    print(pred_classes)

    for i in range(len(boxes)):

        x1, y1, x2, y2 = boxes[i]
        bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]

        rle_mask = encode_mask(pred_masks[i])  # Convert mask to RLE

        detection = {
            "image_id": image_id,
            "category_id": int(pred_classes[i]),
            "bbox": bbox,
            "score": float(scores[i]),
            "segmentation": rle_mask  # Store segmentation in RLE format
        }
        detections.append(detection)

    return detections

def detect_and_save(cfg, image_dir, output_json):
    total_time = 0
    valid_folders = [f"{x:04d}" for x in val]  # Convert validation IDs to string format
    folder_paths = []

    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if os.path.isdir(folder_path) and folder in valid_folders:
            folder_paths.append(folder_path)

    print(folder_paths)
    all_detections = []

    for folder_path in folder_paths:
        print(f"Processing: {folder_path}")

        folder_id = int(os.path.basename(folder_path))

        image_files = sorted(f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png')))
        
        for image_file in image_files:
            print(image_file, folder_path)

            start_time = time.time()

            image_id = int(image_file[:6])

            outputs = run_inference(cfg, os.path.join(folder_path, image_file))

            detections = format_detections(outputs, int(1e5 * folder_id + image_id))
            all_detections.extend(detections)
            
            end_time = time.time()
            total_time += (end_time - start_time)

    average_inference_time = total_time / max(1, len(all_detections))
    
    print(f"Average inference time: {average_inference_time:.2f} seconds per image")
    print(f"Total detections: {len(all_detections)}")

    save_json_safe(all_detections, output_json)

if __name__ == "__main__":
    cfg = setup_cfg()
    
    image_dir = "dataset/image_02/"
    output_json = "det_mask.json"
    
    detect_and_save(cfg, image_dir, output_json)
