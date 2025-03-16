import os
import cv2
import time

from pycocotools import mask as mask_util

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.config import get_cfg
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader


DATA_ROOT = "data/"
TRAIN_JSON = "w2/merged_annotations/fine_tune_train.json"
VAL_JSON = "w2/merged_annotations/fine_tune_val.json"


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
def register_dataset():
    """Registers the dataset in Detectron2's catalog."""
    DatasetCatalog.register(
        "train_dataset", lambda: load_coco_json(TRAIN_JSON, os.path.join(DATA_ROOT, "train"))
    )
    DatasetCatalog.register(
        "val_dataset", lambda: load_coco_json(VAL_JSON, os.path.join(DATA_ROOT, "val"))
    )
    
    # Define class names
    classes = ["class_1", "class_2"]
    MetadataCatalog.get("train_dataset").set(thing_classes=classes)
    MetadataCatalog.get("val_dataset").set(thing_classes=classes)


# Load configuration
def run_finetune_detectron():

    cfg = get_cfg()
    MODELS = {
        "mask_r": "mask_rcnn_R_50_FPN_3x",
        "mask_x": "mask_rcnn_X_101_32x8d_FPN_3x",
        "mask_rdc": "mask_rcnn_R_50_DC5_1x",
    }

    model_name = "mask_rdc"  
    model_path = f"COCO-InstanceSegmentation/{MODELS[model_name]}.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    # Modify for fine-tuning
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Pre-trained weights
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust batch size based on GPU memory
    cfg.SOLVER.BASE_LR = 0.0001  # Lower LR for fine-tuning
    cfg.SOLVER.MAX_ITER = 10000  # Adjust based on dataset size
    cfg.SOLVER.STEPS = []  # No LR decay
    cfg.MODEL.DEVICE = "cuda:0"
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.CHECKPOINT_PERIOD = 8000  # Save model every 2000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Increase for better results
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Change based on your dataset
    cfg.TEST.EVAL_PERIOD = 5000  # Evaluate every 500 iterations
    cfg.OUTPUT_DIR = "output_masks/model_4"

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize trainer
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir="output_masks/model_4")
    val_loader = build_detection_test_loader(cfg, "val_dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == "__main__":


    start_time = time.time()

    register_dataset()
    run_finetune_detectron()


    end_time = time.time()
    inference_time = end_time - start_time
    
    hours = int(inference_time // 3600)
    minutes = int((inference_time % 3600) // 60)
    seconds = inference_time % 60

    print(f"Inference time: {hours}h {minutes}m {seconds:.2f}s")
