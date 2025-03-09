# C5 Project: Multimodal Recognition - Week1

## Tasks Breakdown

### a. Setup and Installation
- Install and configure the required frameworks:
  - Detectron2
  - HuggingFace
  - Ultralytics (for YOLOv8 and above)

### b. Familiarization
- Explore the structure and annotations of the KITTI-MOTS dataset.
- Get acquainted with Detectron2, HuggingFace, and Ultralytics frameworks.

### c. Running Inference
- Use pre-trained models to run inference on the KITTI-MOTS dataset:
  - Faster R-CNN (Detectron2)
  - DeTR (HuggingFace)
  - YOLOv(>8) (Ultralytics)

### d. Model Evaluation
- Evaluate the performance of pre-trained Faster R-CNN, DeTR, and YOLOv(>8) on the KITTI-MOTS dataset.
- Compute AP@0.5 and other relevant metrics.

### e. Fine-tuning on KITTI-MOTS (Similar Domain)
- Fine-tune the following models on the KITTI-MOTS dataset:
  - Faster R-CNN (Detectron2)
  - DeTR (HuggingFace)
  - YOLOv(>8) (Ultralytics)
- Apply data augmentation:
  - Use Albumentations for Faster R-CNN and DeTR.
  - Adjust augmentation settings in YOLO's configuration.

### f. Fine-tuning for Domain Shift
- Fine-tune either Faster R-CNN or DeTR on a different dataset (DeART) to study domain shift effects.

### g. Model Analysis
- Compare the object detection models based on:
  - Number of parameters
  - Inference time
  - Robustness across datasets
  - Overall performance differences

## Setup

Clone the repository:

```bash
git clone https://github.com/grig001/mcv-c5-team8.git
```
Navigate to the corresponding week's folder:
```bash
cd w1
```

## Project Structure
```bash
mcv-c5-team8/ 
│── w1/
│ ├── detectron/
│ ├── HuggingFace/
│ ├── ultralytics/
│ └── README.md
│── README.md
```

## Requirements
- Python 3.x
- Detectron2
- HuggingFace Transformers
- Ultralytics YOLO



## Contributors

- Grigor Grigoryan
- Vincent Heuer
- Philip Zetterberg

