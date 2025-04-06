# Week 4 - Image Captioning

## Folder Structure

```bash
week4/
├── datasets
│   └── Food_Images
│       └── food_partitions.npy
├── task1
│   ├── model_output
│   ├── utils
│   │   ├── __pycache__
│   │   │   └── metrics.cpython-310.pyc
│   │   └── metrics.py
│   ├── evaluate_gpt2_fine_tuned.py
│   ├── evaluate_vit_fine_tuned.py
│   ├── evaluate_vit_gpt2_fine_tuned.py
│   ├── evaluate_vit_gpt2_pretrained.py
│   ├── fine_tune_gpt2_optuna.py
│   ├── fine_tune_gpt2.py
│   ├── fine_tune_vit_gpt2_optuna.py
│   ├── fine_tune_vit_gpt2.py
│   └── fine_tune_vit.py
├── task2
│   ├── model_output
│   ├── models
│   │   └── vit_finetuned_epoch_10
│   ├── evaluate_fine_tuned_llama_1B.py
│   ├── evaluate_fine_tuned_llama_3B.py
│   ├── evaluate_gemma_27B.py
│   ├── fine_tune_llama_1B.py
│   └── fine_tune_llama_3B.py
└── README.md
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

Create a Conda environment named `team8_wee4` and activate it:
```bash
conda create -n team8_wee4 python=3.10 -y
conda activate team8_wee4
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
    transformers evaluate wandb optuna peft
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


## Notes
- The guide sets up **one Conda environment (`team8_wee4`)** for all required models and packages.
- If separate environments are required, create them with:
  ```bash
  conda create -n <env_name> python=3.10 -y
  conda activate <env_name>
  ```
- Always activate the environment before running any scripts:
  ```bash
  conda activate team8_wee4
  ```

---

## Usage - task1

### Change directory
Change directory into task1 with:
```bash
cd task1 
```

### 1. Evaluation of Pretrained Model
To evaluate the model with pretrained weights, you can use the following command. This model is pretrained and can be directly tested for performance:

- **VIT-GPT2 Pretrained Model:**
```bash
python evaluate_vit_gpt2_pretrained.py
```

---

### 2. Fine-Tuning
The training scripts allow you to fine-tune the models on your dataset. By default, all training processes are set to **10 epochs**. If you want to change the number of epochs, simply modify the parameter in the corresponding training script.

The fine-tuned models are automatically saved in the directory: `model_output/{model_name}`.

To fine-tune the models, use the following commands:

- **Fine-Tune VIT-GPT2 Model (No Layers Froozen):**
```bash
python fine_tune_vit_gpt2.py
```

- **Fine-Tune VIT while freezing GPT2:**
```bash
python fine_tune_vit.py
```

- **Fine-Tune GPT2 while freezing VIT:**
```bash
python fine_tune_gpt2.py
```

---

### 3. Fine-Tuning with Optuna Search
Additionally the following python scripts allow you to do an optuna search while fine tuning the models. By default, all training processes are set to **10 epochs**. If you want to change the number of epochs, simply modify the parameter in the corresponding training script.
All the other Hyperparameters will be explored during this search.

To run the optuna search for fine-tuning these models, use the following commands:

- **Optuna search on Fine-Tune of VIT-GPT2 Model (No Layers Froozen):**
```bash
python fine_tune_vit_gpt2_optuna.py
```

- **Optuna search on Fine-Tune of GPT2 while freezing VIT:**
```bash
python fine_tune_gpt2_optuna.py
```

---

### 4. Evaluation of Fine-Tuned Models
To evaluate the models with fine-tuned weights, you can use the following commands:

- **Evaluation of Fine-Tuned VIT-GPT2 Model (No Layers Froozen):**
```bash
python evaluate_vit_gpt2_fine_tuned.py
```

- **Evaluation of Fine-Tuned VIT with froozen GPT2:**
```bash
python evaluate_vit_fine_tuned.py
```

- **Evaluation of Fine-Tuned GPT2 while freezing VIT:**
```bash
python evaluate_gpt2_fine_tuned.py
```

These scripts will automatically access the fine-tuned models from their respective folders under `model_output/{model_name}` and print the evaluation metrics.

---

## Usage - task2

### Change directory
Change directory into task2 with:
```bash
cd task2
```

### 1. Evaluation of Pretrained Model:
To evaluate the model with pretrained weights, you can use the following command. This model is pretrained and can be directly tested for performance:

- **Gemma Pretrained Model:**
```bash
python evaluate_gemma_27B.py
```

---

### 2. Fine-Tuning

For this you need to cp your best VIT model into ./models/ and update the code if necessary:

```bash
cp -r ../task1/model_output/vit_finetuned_epoch_10 ./models/
```

The training scripts allow you to fine-tune the models on your dataset. By default, all training processes are set to **10 epochs**. If you want to change the number of epochs, simply modify the parameter in the corresponding training script.

The fine-tuned models are automatically saved in the directory: `model_output/{model_name}`.

To fine-tune the models, use the following commands:

- **Fine-Tune Llama-3.2-1B with Lora:**
```bash
python fine_tune_llama_1B.py
```

- **Fine-Tune Llama-3.2-3B with Lora:**
```bash
python fine_tune_llama_3B.py
```

---

### 4. Evaluation of Fine-Tuned Models
To evaluate the models with fine-tuned weights, you can use the following commands:

- **Evaluation of Fine-Tuned Llama-3.2-1B (Lora):**
```bash
python evaluate_fine_tuned_llama_1B.py
```

- **Evaluation of Fine-Tuned Llama-3.2-3B (Lora):**
```bash
python evaluate_fine_tuned_llama_3B.py
```

These scripts will automatically access the fine-tuned models from their respective folders under `model_output/{model_name}` and print the evaluation metrics.
