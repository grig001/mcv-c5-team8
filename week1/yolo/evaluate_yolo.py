import os
import numpy as np
import glob
from collections import defaultdict
from sklearn.metrics import average_precision_score

# This script takes the predictions from a pretrained model and evaluates them.

#class mapping
KITTI_CLASSES = {"Car": 1, "Pedestrian": 2}

GT_BBOXES_DIR = "gt_bboxes"  #dir with gt
PREDICTIONS_DIR = "output_yolov8_kitti_mots_txt"  #dir with predictions

def load_gt_bboxes(gt_file):
    """Load ground truth bounding boxes from the new YOLO-like format."""
    gt_bboxes = defaultdict(list)

    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            image_name = parts[0]  
            class_name = parts[1]  
            confidence = float(parts[2])  #always 1.00 in gt
            bbox = list(map(int, parts[3:]))

            frame_id = int(image_name.split(".")[0])
            class_id = KITTI_CLASSES.get(class_name, -1) 
        
            if class_id != -1:
                gt_bboxes[frame_id].append((class_id, bbox))

    return gt_bboxes

def load_predictions(pred_file):
    """Load YOLOv8 predictions."""
    pred_bboxes = defaultdict(list)

    with open(pred_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]  
            class_name = parts[1]
            confidence = float(parts[2])
            bbox = list(map(int, parts[3:]))  

            frame_id = int(image_name.split(".")[0])  
            class_id = KITTI_CLASSES.get(class_name, -1) 
            
            if class_id != -1:
                pred_bboxes[frame_id].append((class_id, confidence, bbox))

    return pred_bboxes

def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_ap(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    """Compute Average Precision (AP) for one class using IoU threshold."""
    y_true = []
    y_scores = []

    for frame_id in sorted(set(gt_bboxes.keys()).union(set(pred_bboxes.keys()))):
        gt_boxes = gt_bboxes.get(frame_id, [])  
        pred_boxes = pred_bboxes.get(frame_id, [])  

        matched_gt = set()
        frame_y_true = []
        frame_y_scores = []

        for class_id, conf, pred_box in sorted(pred_boxes, key=lambda x: -x[1]):
            best_iou = 0
            best_gt_idx = -1

            for i, (gt_class_id, gt_box) in enumerate(gt_boxes):
                if gt_class_id == class_id and i not in matched_gt:
                    iou_score = iou(pred_box, gt_box)
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx != -1:
                frame_y_true.append(1)  
                matched_gt.add(best_gt_idx)
            else:
                frame_y_true.append(0)  

            frame_y_scores.append(conf)

        
        num_false_negatives = len(gt_boxes) - len(matched_gt)
        frame_y_true.extend([1] * num_false_negatives)  
        frame_y_scores.extend([0] * num_false_negatives)  

        y_true.extend(frame_y_true)
        y_scores.extend(frame_y_scores)

    if not y_true or not y_scores:
        return 0

    return average_precision_score(y_true, y_scores)

def evaluate_map():
    """Evaluate mAP across all sequences."""
    ap_per_class = defaultdict(list)

    for gt_file in sorted(glob.glob(os.path.join(GT_BBOXES_DIR, "gt_bboxes_*.txt"))):
        seq_id = os.path.basename(gt_file).split("_")[-1].split(".")[0]
        pred_file = os.path.join(PREDICTIONS_DIR, f"predictions_{seq_id}.txt")

        if not os.path.exists(pred_file):
            print(f"No predictions for sequence {seq_id}, skipping...")
            continue

        print(f"Evaluating Sequence: {seq_id}")

        gt_bboxes = load_gt_bboxes(gt_file)
        pred_bboxes = load_predictions(pred_file)

        for class_name, class_id in KITTI_CLASSES.items():
            ap = compute_ap(
                {k: [b for b in v if b[0] == class_id] for k, v in gt_bboxes.items()},
                {k: [b for b in v if b[0] == class_id] for k, v in pred_bboxes.items()},
            )
            ap_per_class[class_name].append(ap)

            print(f"AP for {class_name}: {ap:.3f}")

    map_per_class = {cls: np.mean(aps) for cls, aps in ap_per_class.items()}
    map_score = np.mean(list(map_per_class.values()))

    print("\nFinal Results:")
    for cls, ap in map_per_class.items():
        print(f"{cls} AP: {ap:.3f}")
    print(f"\nMean Average Precision (mAP): {map_score:.3f}")

#main
evaluate_map()
