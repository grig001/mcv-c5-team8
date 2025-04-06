import os
import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from utils.metrics import compute_metrics  # Importing your metrics function

# print statement to identify job outputs

print("filename: evaluate_vit_gpt2_pretrained.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize Weights & Biases
wandb.init(project="C5_W3", entity="C3_MCV_LGVP")


# Load Data
base_path = '../datasets/Food_Images/'
img_path = f'{base_path}Food_Images/'
cap_path = f'{base_path}Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'
data = pd.read_csv(cap_path)
partitions = np.load(f'{base_path}food_partitions.npy', allow_pickle=True).item()


# Clean Data
dropped_indices = data[data['Title'].isna()].index
partitions['test'] = [idx for idx in partitions['test'] if idx not in dropped_indices]
data = data.dropna(subset=['Title'])

# Reset index after filtering
data = data.reset_index(drop=True)
valid_indices = set(data.index)  # The valid indices after filtering

# Ensure test indices are valid
partitions['test'] = [idx for idx in partitions['test'] if idx in valid_indices]


class CustomDataset(Dataset):
    def __init__(self, data, partition):
        self.data = data.loc[partition].reset_index(drop=True)  # Fix applied here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_name = item['Image_Name'] + '.jpg'
        img_path_full = f'{img_path}{img_name}'

        # Check if the image file exists
        if not os.path.exists(img_path_full):
            # Return None for image and an empty string for caption if file not found
            return None, None  

        try:
            img = Image.open(img_path_full).convert('RGB')  # Load as PIL image
            img = np.array(img)  # Convert PIL image to numpy array
            caption = item['Title']
            return img, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None  # Skip problematic images


# Create Dataloader
test_dataset = CustomDataset(data, partitions['test'])

def collate_fn(batch):
    imgs = []
    captions = []
    for img, caption in batch:
        if img is not None and caption is not None:
            imgs.append(img)
            captions.append(caption)
    return imgs, captions

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


# Load the Model
model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
model.config.pad_token_id = tokenizer.eos_token_id
model.to(DEVICE)


# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for imgs, captions in dataloader:
            if len(imgs) == 0:  # If no valid images in this batch
                continue

            # Feature extractor expects numpy arrays, not PIL images
            pixel_values = feature_extractor(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
            output_ids = model.generate(pixel_values)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            all_predictions.extend(preds)
            all_references.extend([[ref] for ref in captions])

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    # Convert np.float64 to regular floats for wandb logging
    metrics = {k: float(v) if isinstance(v, np.float64) else v for k, v in metrics.items()}
    wandb.log(metrics)  # Log metrics to wandb
    print(metrics)

    return metrics


# Run Evaluation
metrics = evaluate_model(model, test_dataloader)

# Finish W&B
wandb.finish()
