import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MODEL_PATH = "./results_detr_kittimots_batch_size_2_augm/final_model_FULLIMG_AUG"
COCO_ANNOTATIONS = "./coco_annotations.json"
DATA_DIR = "/ghome/c5mcv08/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_PRED_JSON = "./detr_predictions.json"

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {1: "Pedestrian", 2: "Car"}
label2id = {"Pedestrian": 1, "Car": 2}


class COCODatasetEval(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_folder, processor):
        self.coco = COCO(annotation_file)
        self.image_folder = image_folder
        self.processor = processor
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.image_folder, image_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        encoding = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "image_id": image_id
        }


def run_inference(model, processor, dataloader):

    model.eval()
    model.to(DEVICE)

    predictions = []
    print("Running inference on validation set...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            image_ids = batch["image_id"]

            outputs = model(pixel_values=pixel_values)

            for i, image_id in enumerate(image_ids):
                logits = outputs.logits[i]
                bboxes = outputs.pred_boxes[i]
                scores = outputs.logits.softmax(-1)[i, :, :-1].max(dim=-1).values
                labels = outputs.logits.softmax(-1)[i, :, :-1].argmax(dim=-1)

                img_info = dataloader.dataset.coco.imgs[image_id.item()]
                img_width, img_height = img_info["width"], img_info["height"]

                for j in range(len(scores)):
                    if scores[j] > 0.1:
                        box = bboxes[j].cpu().numpy()
                        x_min = int(round(box[0] * img_width))
                        y_min = int(round(box[1] * img_height))
                        width = int(round(box[2] * img_width))
                        height = int(round(box[3] * img_height))

                        predictions.append({
                            "image_id": int(image_id.item()),
                            "category_id": label2id[id2label[labels[j].item() + 1]],
                            "bbox": [x_min, y_min, width, height],
                            "score": round(float(scores[j]), 2)
                        })

    with open(OUTPUT_PRED_JSON, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {OUTPUT_PRED_JSON}")
    return OUTPUT_PRED_JSON


def evaluate_coco(gt_json, pred_json):

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


def main():

    processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

    dataset = COCODatasetEval(COCO_ANNOTATIONS, DATA_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: {
        "pixel_values": torch.stack([item["pixel_values"] for item in x]),
        "image_id": torch.tensor([item["image_id"] for item in x])
    })

    pred_json = run_inference(model, processor, dataloader)
    evaluate_coco(COCO_ANNOTATIONS, pred_json)


if __name__ == "__main__":
    main()
