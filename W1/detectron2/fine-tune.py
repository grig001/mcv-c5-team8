import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os

import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json

MODELS = {
    "retina": "retinanet_R_50_FPN_3x",
    "faster": "faster_rcnn_R_50_FPN_3x",
}

# Dataset paths
DATA_ROOT = "data"
TRAIN_JSON = "dataset/merged_annotations/train_fine_tune.json"
VAL_JSON = "dataset/merged_annotations/val_fine_tune.json"

def register_dataset():
    """Registers the dataset in Detectron2's catalog."""
    DatasetCatalog.register(
        "train_dataset", lambda: load_coco_json(TRAIN_JSON, os.path.join(DATA_ROOT, "train"))
    )
    DatasetCatalog.register(
        "val_dataset", lambda: load_coco_json(VAL_JSON, os.path.join(DATA_ROOT, "val"))
    )
    
    # Define class names
    classes = ["class_1", "class_10"]
    MetadataCatalog.get("train_dataset").set(thing_classes=classes)
    MetadataCatalog.get("val_dataset").set(thing_classes=classes)

def run_finetune_detectron():
    """Sets up and trains a Faster R-CNN model using Detectron2."""
    
    model_name = "faster"  # Change to "retina" if needed
    model_path = f"COCO-Detection/{MODELS[model_name]}.yaml"

    # Configuration setup
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    
    # Fine-tuning settings
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)  # Pretrained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # Confidence threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.25   # Non-max suppression threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2         # "car" and "bike"

    # Dataset assignments
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)

    # Training settings
    cfg.SOLVER.IMS_PER_BATCH = 1  # Adjust based on GPU memory
    cfg.SOLVER.BASE_LR = 1e-4  # Learning rate
    cfg.SOLVER.MAX_ITER = 3000    # Iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoints every 1000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # RoI batch size

    # Device setup (Multi-GPU support)
    cfg.MODEL.DEVICE = "cuda:0"

    # Output directory
    cfg.OUTPUT_DIR = f"models/faster_3/"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Trainer initialization
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    register_dataset()
    run_finetune_detectron()
