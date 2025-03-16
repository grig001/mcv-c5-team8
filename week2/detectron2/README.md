## Folder Structure

```bash
Week2/detectron/
│-- scripts/
│   ├── retrieve_bbox_gt.py  # Extracts bounding boxes and masks
│   ├── convert_masks.py     # Converts to COCO format
│   ├── merge.py             # Merges and splits datasets
│   ├── mask_detection.py    # Runs segmentation detection
│   ├── fine-tune.py         # Fine-tunes the model and evaluates
│-- README.md
```


## Installation Steps
1. Clone the repository:
```bash
https://github.com/grig001/mcv-c5-team8.git
```
2. Navigate to the corresponding folder:
```bash
cd week2/detectron2
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
