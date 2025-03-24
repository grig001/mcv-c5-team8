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


## Usage

### Usage (Using the Notebook)

You can try out the code by simply opening the notebook `week3.ipynb`. It contains all the necessary code cells that you need to run. Just execute them in order and follow the comments provided within the notebook. This provides a way to run the training, fine-tuning, and evaluation processes without needing to use the terminal commands directly.
Be aware of the file paths in the beginning of the notebook - they might not be correct.

---

## Usage (Using Scripts)

### 1. Creating Partitions
If you want to create new partitions for your dataset, you can do so by running:
```bash
python utils/create_food_partitions.py
```
This script generates the necessary partition files which will be stored in the `datasets/Food_Images` folder.

---

### 2. Inference using Pretrained Models
To evaluate the models with pretrained weights, you can use the following commands. These models are pretrained and can be directly tested for performance:

- **Baseline Model (Simple CNN):**
```bash
python evaluate_pretrained_baseline.py
```

- **LSTM Decoder Model:**
```bash
python evaluate_pretrained_lstm.py
```

- **VGG19 Encoder-Decoder Model:**
```bash
python evaluate_pretrained_vgg19.py
```

- **VGG19 + LSTM Encoder-Decoder Model:**
```bash
python evaluate_pretrained_vgg19_lstm.py
```

---

### 3. Fine-Tuning
The training scripts allow you to fine-tune the models on your dataset. By default, all training processes are set to **10 epochs**. If you want to change the number of epochs, simply modify the parameter in the corresponding training script.

The fine-tuned models are automatically saved in the directory: `models/{model_name}`.

To fine-tune the models, use the following commands:

- **Baseline Model:**
```bash
python train_baseline.py
```

- **LSTM Decoder Model:**
```bash
python train_lstm.py
```

- **VGG19 Model:**
```bash
python train_vgg19.py
```

- **VGG19 with Frozen Weights:**
```bash
python train_vgg19_frozen_weights.py
```

- **VGG19 + LSTM Model:**
```bash
python train_vgg19_lstm.py
```

---

### 4. Inference using Fine-Tuned Models
Once fine-tuning is complete, you can evaluate the fine-tuned models by running the following commands:

- **Baseline Model:**
```bash
python evaluate_fine_tuned_baseline.py
```

- **LSTM Decoder Model:**
```bash
python evaluate_fine_tuned_lstm.py
```

- **VGG19 Model:**
```bash
python evaluate_fine_tuned_vgg19.py
```

- **VGG19 with Frozen Weights:**
```bash
python evaluate_fine_tuned_vgg19_frozen_weights.py
```

- **VGG19 + LSTM Model:**
```bash
python evaluate_fine_tuned_vgg19_lstm.py
```

These scripts will automatically access the fine-tuned models from their respective folders under `models/{model_name}` and print the evaluation metrics.

---

### 5. Jobs
In Addition to that you can also run all the scripts with a sbatch job command. For this you only need to change the job file, which is in the source folder.  

