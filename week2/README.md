# Week 2 - Object Segmentation

## Folder Structure

```bash
Week2/

│-- README.md
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
Install Detectron2 with CUDA 12.1 support.
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.5/index.html
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
python
```

### detectron2 - Mask R-CNN

### huggingface - mask2former

### ultralytics - yolov8n-seg


## Tasks Breakdown
