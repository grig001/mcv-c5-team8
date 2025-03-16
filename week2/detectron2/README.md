


### Extract ground truth bounding boxes and masks
```bash
python retrieve_bbox_gt.py
```

### Convert to COCO format
```bash
python convert_masks.py
```

### Merge datasets and split into train/val
```bash
python merge.py
```

### Run segmentation detection
```bash
python mask_detection.py
```

## Dependencies
Ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- OpenCV
- Detectron2
