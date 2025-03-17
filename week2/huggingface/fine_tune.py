import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation, Trainer, TrainingArguments
from pycocotools import mask as mask_utils


# Load COCO-style annotations
def load_annotations(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


# Custom Dataset without Augmentations
class Mask2FormerDataset(Dataset):
    def __init__(self, coco_json, processor):
        self.images = coco_json["images"]
        self.annotations = coco_json["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in coco_json["categories"]}
        self.processor = processor

        # Index annotations by image_id
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_meta = self.images[idx]
        image_path = img_meta["image"]
        image_id = img_meta["id"]
        width, height = img_meta["width"], img_meta["height"]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        anns = self.image_id_to_annotations.get(image_id, [])

        masks = np.zeros((len(anns), height, width), dtype=np.uint8)
        class_labels = []

        for i, ann in enumerate(anns):
            category_id = ann["category_id"]
            rle_segmentation = ann["segmentation"]

            if isinstance(rle_segmentation, dict) and "counts" in rle_segmentation:
                binary_mask = mask_utils.decode(rle_segmentation)
                masks[i] = binary_mask
                class_labels.append(category_id)

        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        masks = torch.tensor(masks, dtype=torch.float32)
        class_labels = torch.tensor(class_labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "mask_labels": masks,
            "class_labels": class_labels
        }


# Load Model & Processor
model_name = "facebook/mask2former-swin-small-coco-instance"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)

# Load Data
train_data = load_annotations("datasets/train_gt.json")
val_data = load_annotations("datasets/val_gt.json")

# Initialize Datasets
train_dataset = Mask2FormerDataset(train_data, processor)
val_dataset = Mask2FormerDataset(val_data, processor)

# Fix Model Weights Mismatch
num_classes = len(train_data["categories"])
model = AutoModelForUniversalSegmentation.from_pretrained(
    model_name,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)


# Data Collation Function
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in batch]),
        "mask_labels": [example["mask_labels"] for example in batch],
        "class_labels": [example["class_labels"] for example in batch]
    }


training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    dataloader_num_workers=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
    fp16=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=collate_fn
)

trainer.train()

trainer.save_model("model_fine_tuned")
processor.save_pretrained("model_fine_tuned")
