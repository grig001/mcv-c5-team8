import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

# Load the Mask2Former model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-small-coco-instance"

processor = AutoImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
model.eval()

# Load evaluation dataset
eval_json_path = "datasets/eval_gt.json"
with open(eval_json_path, "r") as f:
    eval_data = json.load(f)

image_list = eval_data["images"]

# Category mappings from the ground truth
category_mapping = {2: 0, 11: 1}

coco_predictions = []


def rle_encode(binary_mask):
    encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


# Run inference
for img_meta in tqdm(image_list, desc="Processing images"):
    image_path = img_meta["image"]
    image_id = img_meta["id"]

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_instance_map = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]

    for segment in pred_instance_map["segments_info"]:
        seg_id = segment["id"]
        predicted_category_id = segment["label_id"]
        score = segment["score"]

        category_id = category_mapping.get(predicted_category_id, predicted_category_id)

        seg_mask = pred_instance_map["segmentation"].cpu().numpy() == seg_id
        encoded_rle = rle_encode(seg_mask)

        ys, xs = np.where(seg_mask)
        if len(xs) > 0 and len(ys) > 0:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]
        else:
            continue

        # Store results
        coco_predictions.append({
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": encoded_rle,
            "bbox": bbox,
            "score": float(score)
        })

# Save predictions to JSON
predictions_json_path = "predictions/predictions_pretrained.json"
with open(predictions_json_path, "w") as f:
    json.dump(coco_predictions, f, indent=4)

print(f"Predictions saved to {predictions_json_path}")

# Run COCO Evaluation
coco_gt = COCO(eval_json_path)
coco_dt = coco_gt.loadRes(predictions_json_path)

coco_eval = COCOeval(coco_gt, coco_dt, "segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
