# Week 3 - Image Captioning

## Folder Structure

```bash
├── datasets
│   ├── Flickr_8k
│   │   ├── captions.txt
│   │   ├── flickr8k_partitions.npy
│   │   └── Images
│   └── Food_Images
│       ├── Food_Images
│       ├── Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv
│       └── food_partitions.npy
├── job_outputs
├── models
│   ├── baseline
│   ├── lstm
│   ├── vgg19
│   ├── vgg19_frozen_weights
│   └── vgg19_lstm
├── utils
│   ├── create_flickr8k_partitions.py
│   ├── create_food_partitions.py
│   └── test_metrics.py
├── baseline_model.py
├── evaluate_fine_tuned_baseline.py
├── evaluate_fine_tuned_lstm.py
├── evaluate_fine_tuned_vgg19_frozen_weights.py
├── evaluate_fine_tuned_vgg19_lstm.py
├── evaluate_fine_tuned_vgg19.py
├── evaluate_pretrained_baseline.py
├── evaluate_pretrained_lstm.py
├── evaluate_pretrained_vgg19_lstm.py
├── evaluate_pretrained_vgg19.py
├── job
├── README.md
├── results.txt
├── train_baseline.py
├── train_lstm.py
├── train_vgg19_frozen_weights.py
├── train_vgg19_lstm.py
├── train_vgg19.py
└──  week3.ipynb
```


## Installation Steps

### 1. Clone the Repository

Clone the repository from GitHub:
```bash
https://github.com/grig001/mcv-c5-team8.git
```

Navigate to the corresponding folder:
```bash
cd week3/
```

### 2. Download the Dataset

Download the dataset from Kaggle:

[Food Ingredients and Recipe Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images?resource=download-directory)

Unzip the downloaded files and store them in the following structure:
```
datasets/Food_Images/Food_Images

datasets/Food_Images/Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv
```
If the folder names differ, make sure to:
- Rename "Food Images" to "Food_Images"
- Rename "Food Ingredients and Recipe Dataset with Image Name Mapping.csv" to "Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv"


### 3. Create a Conda Environment

Create a Conda environment named `team8_week3` and activate it:
```bash
conda create -n team8_week3 python=3.10 -y
conda activate team8_week3
```


### 4. Install PyTorch with CUDA Support

Install PyTorch, TorchVision, and TorchAudio with CUDA 12.1 support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```


### 5. Install Required Python Packages

Install all necessary dependencies using `pip`:
```bash
pip install --upgrade pip
pip install numpy pandas scipy matplotlib opencv-python pycocotools \
    transformers evaluate
```


### 6. Install TorchVision Models

Install `torchvision` separately to ensure compatibility:
```bash
pip install torchvision
```


### 7. Check Dependencies

To confirm successful installation, run the following commands:

#### Check PyTorch and CUDA
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### Check Transformers
```bash
python -c "from transformers import ResNetModel; print('Transformers installed successfully')"
```

#### Check Evaluate
```bash
python -c "import evaluate; print('Evaluate library installed successfully')"
```


### 8. Test Metrics
```bash
python utils/test_merics.py
```


## Notes
- The guide sets up **one Conda environment (`team8_week3`)** for all required models and packages.
- If separate environments are required, create them with:
  ```bash
  conda create -n <env_name> python=3.10 -y
  conda activate <env_name>
  ```
- Always activate the environment before running any scripts:
  ```bash
  conda activate team8_week3
  ```


## Usage

Make sure your dataset is properly structured before running any scripts.

You can proceed with your training, evaluation, and other tasks once the environment is set up and dependencies are installed successfully.


