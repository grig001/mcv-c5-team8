import os
import json
import torch
import numpy as np
from collections import defaultdict
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_BBOXES_DIR = "gt_bboxes_detr"
PRED_BBOXES_DIR = "output_detr_training_gt_format"
OUTPUT_COCO_JSON = "detr_coco_eval.json"

KITTI_TO_COCO = {"Pedestrian": 1, "Car": 3}


def convert_to_coco(gt_dir, pred_dir, output_json):
    """Converts KITTI-MOTS format into COCO JSON for evaluation."""
    coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "Pedestrian"}, {"id": 3, "name": "Car"}]}
    annotation_id = 1
    image_id_map = {}

    for gt_file in sorted(os.listdir(gt_dir)):
        seq_id = gt_file.split("_")[-1].split(".")[0]
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, f"predictions_{seq_id}.txt")

        if not os.path.exists(pred_path):
            continue

        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_name, class_name, conf, x_min, y_min, x_max, y_max = parts

                if class_name not in KITTI_TO_COCO:
                    continue

                if image_name not in image_id_map:
                    img_id = len(image_id_map)
                    image_id_map[image_name] = img_id
                    coco_data["images"].append({"id": img_id, "file_name": image_name, "height": 375, "width": 1242})

                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                width, height = x_max - x_min, y_max - y_min

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id_map[image_name],
                    "category_id": KITTI_TO_COCO[class_name],
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1

    predictions = []
    for pred_file in sorted(os.listdir(pred_dir)):
        seq_id = pred_file.split("_")[-1].split(".")[0]
        pred_path = os.path.join(pred_dir, pred_file)

        with open(pred_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_name, class_name, conf, x_min, y_min, x_max, y_max = parts
                if class_name not in KITTI_TO_COCO or image_name not in image_id_map:
                    continue

                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                width, height = x_max - x_min, y_max - y_min

                predictions.append({
                    "image_id": image_id_map[image_name],
                    "category_id": KITTI_TO_COCO[class_name],
                    "bbox": [x_min, y_min, width, height],
                    "score": float(conf)
                })

    with open(output_json, "w") as f:
        json.dump({"annotations": coco_data["annotations"], "images": coco_data["images"], "categories": coco_data["categories"]}, f, indent=4)

    pred_json = output_json.replace(".json", "_preds.json")
    with open(pred_json, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"COCO JSON saved: {output_json}")
    print(f"Prediction JSON saved: {pred_json}")
    return output_json, pred_json


def evaluate_coco(gt_json, pred_json):
    """Runs COCO evaluation using Detectron2 metrics."""
    coco_gt = COCO(gt_json)
    coco_pred = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "mAP@0.5:0.95": coco_eval.stats[0],
        "mAP@0.50": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
        "AP@0.5:0.95 (Small)": coco_eval.stats[3],
        "AP@0.5:0.95 (Medium)": coco_eval.stats[4],
        "AP@0.5:0.95 (Large)": coco_eval.stats[5],
        "AR@0.5:0.95": coco_eval.stats[6]
    }

    print("\nFinal COCO Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")


gt_json, pred_json = convert_to_coco(GT_BBOXES_DIR, PRED_BBOXES_DIR, OUTPUT_COCO_JSON)
evaluate_coco(gt_json, pred_json)
