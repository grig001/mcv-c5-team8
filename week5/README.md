# Week 5 - Diffusion models

## Folder Structure

```bash
./week5
├── datasets
│   └── Food_Images
│       ├── Food_Images
│       ├── Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv
│       ├── food_partitions.npy
│       ├── food_partitions_only_synthetic.npy
│       ├── food_partitions_plus_10pct_synthetic.npy
│       ├── food_partitions_plus_20pct_synthetic.npy
│       ├── food_partitions_plus_30pct_synthetic.npy
│       ├── food_partitions_plus_40pct_synthetic.npy
│       ├── food_partitions_plus_50pct_synthetic.npy
│       └── synthetic_food_partitions.npy
├── prompts
│   ├── dataset_likely_all_5400.csv
│   └── first_100.txt
├── evaluate_across_all_parameters.py
├── evaluate_all_partitions.py
├── fine_tune_vit_gpt2_all_partitions.py
├── generate_first_100_images.py
├── generate_images_with_xl_turbo.py
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
cd week5/
```

### 2. Download the Dataset

Download the dataset from Kaggle:

[Food Ingredients and Recipe Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images?resource=download-directory)

Unzip the downloaded files and store ONLY the images in the following folder:
```
datasets/Food_Images/Food_Images
```

Delete the following csv file:

```
Food Ingredients and Recipe Dataset with Image Name Mapping.csv
```


### 3. Create a Conda Environment

Create a Conda environment named `team8_week5` and activate it:
```bash
conda create -n team8_week5 python=3.10 -y
conda activate team8_week5
```


## 4. Install PyTorch with CUDA Support

Install PyTorch, TorchVision, and TorchAudio with CUDA 12.1 support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```


## 5. Install Required Python Packages

Install core dependencies:

```bash
pip install --upgrade pip
pip install numpy pandas tqdm scikit-learn opencv-python \
    transformers evaluate diffusers accelerate peft \
    matplotlib scipy Pillow
```


## 6. Install Scheduler Support for Diffusers

Enable scheduler options like DDIM, DDPM, and DPM Solver:

```bash
pip install diffusers[torch]
```


## 7. Install Weights & Biases (Optional)

Used for experiment tracking (currently commented out in the scripts):

```bash
pip install wandb
```


## 8. Validate Installations

### Check PyTorch and CUDA:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Check Diffusers and Transformers:
```bash
python -c "from diffusers import StableDiffusionPipeline; print('Diffusers OK')"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
```

### Check Evaluate:
```bash
python -c "import evaluate; print('Evaluate OK')"
```

### Check wandb (optional):
```bash
python -c "import wandb; print('wandb OK')"
```


## Notes
- The guide sets up **one Conda environment (`team8_week5`)** for all required models and packages.
- If separate environments are required, create them with:
  ```bash
  conda create -n <env_name> python=3.10 -y
  conda activate <env_name>
  ```
- Always activate the environment before running any scripts:
  ```bash
  conda activate team8_week5
  ```


## Usage

### Change directory
Change into the project folder:
```bash
cd week5
```

## Task a: Generate First 100 Images

Generate the first 100 images from a list of prompts across four Stable Diffusion models:

```bash
python generate_first_100_images.py
```

This will create subdirectories under `generated_images/first_100/` for each model variant.


## Task b: Evaluate All Parameter Combinations

Generate images using various configurations of models, samplers, CFG scales, and prompt variations:

```bash
python evaluate_across_all_parameters.py
```

Images will be saved to `task_b_experiments/food/` following a structured folder layout.


## Task d: Generate XL-Turbo Images from CSV

Use SDXL Turbo to generate images from prompts defined in a CSV file using both default and optimized parameters:

```bash
python generate_images_with_xl_turbo.py
```

Generated images will be saved in:
- `generated_images/dataset_likely/default/`
- `generated_images/dataset_likely/optimized/`


## Task e: Fine-Tune VIT-GPT2 on All Partitions

Fine-tune the VisionEncoderDecoder model across different dataset splits. Models are saved automatically by epoch:

```bash
python fine_tune_vit_gpt2_all_partitions.py
```

Fine-tuned models are saved to `model_output/{partition}_epoch{n}`.


## Task e: Evaluate Fine-Tuned Models on All Partitions

Evaluate each model saved from the fine-tuning task and log performance metrics:

```bash
python evaluate_all_partitions.py
```

Evaluation results are printed to the console and optionally logged to Weights & Biases.


## References

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Evaluate Library](https://huggingface.co/docs/evaluate)
- [Weights & Biases](https://wandb.ai/)
- [PyTorch](https://pytorch.org/)
- [Stable Diffusion Models by Stability AI](https://huggingface.co/stabilityai)

