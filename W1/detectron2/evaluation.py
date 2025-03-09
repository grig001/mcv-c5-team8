import json

def compute_iou(pred_bbox, gt_bbox):

    x1, y1, w1, h1 = pred_bbox
    x2, y2, w2, h2 = gt_bbox
    
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    
    pred_area = w1 * h1
    gt_area = w2 * h2
    
    union_area = pred_area + gt_area - intersection_area
    
    return intersection_area / union_area


def match_predictions_to_gt(predictions, ground_truths, iou_threshold=0.5):
    matches = []
    
    for pred in predictions:
        best_iou = 0
        best_gt = None
        for gt in ground_truths:
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        if best_iou >= iou_threshold:
            matches.append((pred, best_gt))
    
    return matches


def evaluate_matches(predictions, ground_truths, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt = set()
    
    matches = match_predictions_to_gt(predictions, ground_truths, iou_threshold)
    
    for pred, gt in matches:
        if gt['id'] not in matched_gt:
            tp += 1
            matched_gt.add(gt['id'])
    
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    return precision, recall


def calculate_ap(precision, recall):
    ap = 0.0
    for i in range(1, len(precision)):
        ap += precision[i] * (recall[i] - recall[i - 1])
    return ap


def evaluate_map(predictions, ground_truths, iou_threshold=0.5):
    all_precisions = []
    all_recalls = []
    
    categories = list(set([pred['category_id'] for pred in predictions] +
                          [gt['category_id'] for gt in ground_truths]))
    
    for category in categories:
        class_predictions = [p for p in predictions if p['category_id'] == category]
        class_gt = [gt for gt in ground_truths if gt['category_id'] == category]
        
        precision, recall = evaluate_matches(class_predictions, class_gt, iou_threshold)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        
        ap = calculate_ap(all_precisions, all_recalls)
        
    return sum(all_precisions) / len(all_precisions) if len(all_precisions) > 0 else 0


def load_annotations(file_path):
    with open(file_path) as f:
        return json.load(f)



gt_file_path = "dataset/merged_annotations/val.json"
pred_file_path = "det_faster_rcnn_fine_tune_faster_2.json"
'''
# Mean Average Precision (mAP) at IoU threshold 0.5: 0.02596119295724039
Mean Average Precision (mAP) at IoU threshold 0.75: 0.013384836507366151
'''


# pred_file_path ='det_faster_rcnn_fine_tune_faster_0.json'
'''
Mean Average Precision (mAP) at IoU threshold 0.5: 0.02750712927756654
Mean Average Precision (mAP) at IoU threshold 0.75: 0.013486216730038024
'''

# pred_file_path = "det_faster_rcnn_fine_tune_faster_1.json"
'''
Mean Average Precision (mAP) at IoU threshold 0.5: 0.01488095238095238
Mean Average Precision (mAP) at IoU threshold 0.75: 0.0005175983436853002
'''

# pred_file_path = "det_faster_rcnn_fine_tune_faster_2.json"
'''
Mean Average Precision (mAP) at IoU threshold 0.5: 0.016689607708189953
Mean Average Precision (mAP) at IoU threshold 0.75: 0.004793039032543506
'''
ground_truths = load_annotations(gt_file_path)['annotations']
predictions = load_annotations(pred_file_path)

# Evaluate mAP
iou_threshold = 0.75
mAP = evaluate_map(predictions, ground_truths, iou_threshold)

print(f"Mean Average Precision (mAP) at IoU threshold {iou_threshold}: {mAP}")
