import json


GT_FILE = "coco_annotations.json"
PRED_FILE = "detr_predictions.json"


def compute_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    iou = inter_area / (area_box1 + area_box2 - inter_area + 1e-6)
    return iou


with open(GT_FILE, "r") as f:
    gt_data = json.load(f)

with open(PRED_FILE, "r") as f:
    pred_data = json.load(f)

gt_boxes = {}
for ann in gt_data["annotations"]:
    image_id = ann["image_id"]
    if image_id not in gt_boxes:
        gt_boxes[image_id] = []
    gt_boxes[image_id].append((ann["bbox"], ann["category_id"]))

pred_boxes = {}
for pred in pred_data:
    image_id = pred["image_id"]
    if image_id not in pred_boxes:
        pred_boxes[image_id] = []
    pred_boxes[image_id].append((pred["bbox"], pred["category_id"], pred["score"]))

for image_id in pred_boxes:
    pred_boxes[image_id].sort(key=lambda x: x[2], reverse=True)

iou_threshold = 0.5
all_precisions = []

for image_id in gt_boxes:
    gt_list = gt_boxes.get(image_id, [])
    pred_list = pred_boxes.get(image_id, [])

    matched_gts = set()
    tp = 0
    fp = 0
    fn = len(gt_list)

    for pred_bbox, pred_class, pred_score in pred_list:
        best_iou = 0
        best_gt_index = -1

        for i, (gt_bbox, gt_class) in enumerate(gt_list):
            if gt_class == pred_class:
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = i

        if best_iou >= iou_threshold and best_gt_index not in matched_gts:
            tp += 1
            fn -= 1
            matched_gts.add(best_gt_index)
        else:
            fp += 1

    precision = tp / (tp + fp + 1e-6)
    all_precisions.append(precision)

mAP_50 = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0

print(f"mAP@0.5: {mAP_50:.3f}")
