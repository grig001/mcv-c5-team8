# Week 2 - Object Segmentation

## Folder Structure

```bash
week2/
├── data_preprocessing
│   └── make_train_val_split.py
├── detectron2
│   ├── README.md
│   └── scripts
│       ├── convert_masks.py
│       ├── fine_tune.py
│       ├── mask_detection.py
│       ├── merge.py
│       └── retrieve_bbox_gt.py
├── domain_shift
│   ├── convert_file_domain_shift.py
│   └── fine-tune_domain_shift.py
├── huggingface
│   ├── datasets
│   │   ├── eval_gt.json
│   │   ├── train_gt_all.json
│   │   ├── train_gt.json
│   │   └── val_gt.json
│   ├── evaluate_coco_files.py
│   ├── fine_tune.py
│   ├── inference_pretrained.py
│   ├── job_mask2former
│   ├── job_output
│   │   └── job_example
│   └── predictions
│       └── predictions_pretrained.json
├── README.md
└── ultralytics
    ├── datasets
    │   ├── yolo_fine_tune.yaml
    │   └── yolo_pretrained.yaml
    ├── evaluate_fine_tuned.py
    ├── fine_tune_augm.py
    ├── fine_tune_optuna.py
    ├── fine_tune.py
    ├── inference_pretrained.py
    ├── job_output
    │   └── job_example
    ├── job_yolo
    └── prepare_yolo_data.py
```


## Installation Steps
1. Clone the repository:
```bash
https://github.com/grig001/mcv-c5-team8.git
```
2. Navigate to the corresponding folder:
```bash
cd week2/
```

## Envirement Setup 

### 1. Create a Conda Environment
Create a Conda environment named `team8_env` and activate it.

```bash
conda create -n team8_env python=3.10 -y
conda activate team8_env
```

### 2. Install PyTorch with CUDA Support
Install PyTorch, TorchVision, and TorchAudio with CUDA 12.1 support.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install Required Python Packages
Install the necessary dependencies using `pip`.

```bash
pip install --upgrade pip
pip install numpy pandas scipy matplotlib opencv-python pycocotools ultralytics transformers
```

### 4. Install Model-Specific Dependencies

#### Mask2Former Dependencies
```bash
pip install albumentations datasets huggingface-hub tqdm
```

#### YOLOv8 Dependencies
```bash
pip install ultralytics
```

#### Detectron2 Dependencies

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 5. Check Dependencies

#### Check YOLOv8
```bash
python -c "import ultralytics; print(f'Ultralytics Version: {ultralytics.__version__}')"
```

#### Check Detectron2
```bash
python -c "import detectron2; print('Detectron2 installed successfully')"
```

#### Check Mask2Former Dependencies
```bash
python -c "import albumentations, datasets, transformers; print('Mask2Former dependencies installed successfully')"
```


## Notes
- This guide sets up **one Conda environment (`team8_env`)** for all models.
- If you need separate environments, create them with:
  ```bash
  conda create -n <env_name> python=3.10 -y
  conda activate <env_name>
  ```
- Always activate the environment before running models:
  ```bash
  conda activate team8_env
  ```



## Usage

Preprocess the data for ultralytics and huggingface: 
```bash
python data_preprocessing/make_train_val_split.py
```

### detectron2 - Mask R-CNN
The README for detectron2 can be found [here](./detectron2/README.md).  

### huggingface - mask2former
The README for detectron2 can be found [here](./huggingface/README.md).  

### ultralytics - yolov8n-seg
The README for detectron2 can be found [here](./ultralytics/README.md).  
