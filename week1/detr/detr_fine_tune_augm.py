import os
import json
import torch
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor, Trainer, TrainingArguments
from pycocotools.coco import COCO

DATA_DIR = "/ghome/c5mcv08/mcv/datasets/C5/KITTI-MOTS/training/image_02"
COCO_ANNOTATIONS = "./coco_annotations.json"
OUTPUT_DIR = "./results_detr_kittimots_batch_size_10_augm"
MODEL_NAME = "facebook/detr-resnet-50"

id2label = {1: "Pedestrian", 2: "Car"}
label2id = {"Pedestrian": 1, "Car": 2}

train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.HueSaturationValue(hue_shift_limit=0.05, sat_shift_limit=0.05, val_shift_limit=0.05, p=0.2),
        A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3),], p=0.2),
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_id'], min_area=1.0, min_visibility=0.1))

val_transform = A.Resize(480, 640, interpolation=cv2.INTER_LINEAR)


class COCODataset(Dataset):
    def __init__(self, annotation_file, image_folder, processor, transform=None):
        self.coco = COCO(annotation_file)
        self.image_folder = image_folder
        self.processor = processor
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.image_folder, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        if self.transform:
            transformed = self.transform(image=image_np)
            image_transformed = transformed["image"]
        else:
            image_transformed = image_np

        image_transformed = Image.fromarray(image_transformed)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        formatted_annotations = []
        for ann in annotations:
            formatted_annotations.append({
                "bbox": ann["bbox"], 
                "category_id": ann["category_id"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"]
            })

        target = {
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "annotations": formatted_annotations
        }

        encoding = self.processor(images=image_transformed, annotations=target, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0]
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def train_detr():
    if not os.path.exists(COCO_ANNOTATIONS):
        raise FileNotFoundError("COCO-Annotationsdatei nicht gefunden. Stelle sicher, dass detr_convert_kitti_coco.py ausgeführt wurde.")

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    dataset = COCODataset(COCO_ANNOTATIONS, DATA_DIR, processor, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        learning_rate=1e-5,
        weight_decay=1e-4,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("Starte Training...")
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{OUTPUT_DIR}/final_model")

    print(f"Fine-Tuned Model gespeichert unter: {OUTPUT_DIR}/final_model")


train_detr()
