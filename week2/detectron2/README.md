## Folder Structure

```bash
detectron2/
├── README.md
└── scripts
    ├── convert_masks.py
    ├── fine_tune.py
    ├── mask_detection.py
    ├── merge.py
    └── retrieve_bbox_gt.py
```


## Usage

Extract ground truth bounding boxes and masks
```bash
python retrieve_bbox_gt.py
```

Convert to COCO format
```bash
python convert_masks.py
```

Merge datasets and split into train/val
```bash
python merge.py
```

Run segmentation detection
```bash
python mask_detection.py
```

Fine-tune the model
```bash
python fine-tune.py
```

## Dependencies
Ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- OpenCV
- Detectron2
- pycocotools
- json
