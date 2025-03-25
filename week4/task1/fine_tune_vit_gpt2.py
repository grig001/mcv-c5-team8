import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AdamW
from tqdm import tqdm
import os
import evaluate

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize W&B
wandb.init(project="C5_W3", entity="C3_MCV_LGVP")

# Metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# Load Data
base_path = '../datasets/Food_Images/'
img_path = f'{base_path}Food_Images/'
cap_path = f'{base_path}Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'
data = pd.read_csv(cap_path)
partitions = np.load(f'{base_path}food_partitions.npy', allow_pickle=True).item()

# Clean Data
dropped_indices = data[data['Title'].isna()].index
partitions['train'] = [idx for idx in partitions['train'] if idx not in dropped_indices]
partitions['valid'] = [idx for idx in partitions['valid'] if idx not in dropped_indices]
data = data.dropna(subset=['Title'])
data = data.reset_index(drop=True)
valid_indices = set(data.index)
partitions['train'] = [idx for idx in partitions['train'] if idx in valid_indices]
partitions['valid'] = [idx for idx in partitions['valid'] if idx in valid_indices]


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, partition, tokenizer, feature_extractor):
        self.data = data.loc[partition].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_name = item['Image_Name'] + '.jpg'
        img_path_full = f'{img_path}{img_name}'

        if not os.path.exists(img_path_full):
            return None, None, None

        try:
            img = Image.open(img_path_full).convert('RGB')
            img = self.feature_extractor(images=img, return_tensors="pt").pixel_values[0]

            caption = item['Title']
            labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", max_length=64, truncation=True).input_ids[0]
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding

            return img, labels, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None, None


def collate_fn(batch):
    imgs, labels, captions = zip(*[(img, label, caption) for img, label, caption in batch if img is not None])
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, captions


# Load Model & Tokenizer
model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
model.config.pad_token_id = tokenizer.pad_token_id
model.to(DEVICE)


# Training Function
def train_one_epoch(model, dataloader, optimizer, step_log=100):
    model.train()
    total_loss = 0

    for step, (imgs, labels, _) in enumerate(tqdm(dataloader, desc="Training")):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(pixel_values=imgs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if step % step_log == 0:
            wandb.log({"Training Loss": loss.item()})

    return total_loss / len(dataloader)


# Evaluation Function
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for imgs, labels, captions in tqdm(dataloader, desc="Validation"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = model.generate(imgs)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_predictions.extend(preds)
            all_references.extend([[ref] for ref in captions])

    metrics = {
        "Validation Loss": total_loss / len(dataloader),
        "BLEU-1": bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"],
        "BLEU-2": bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"],
        "ROUGE-L": rouge.compute(predictions=all_predictions, references=all_references)["rougeL"],
        "METEOR": meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
    }
    wandb.log(metrics)
    return metrics


# Data Loaders
train_dataset = CustomDataset(data, partitions['train'], tokenizer, feature_extractor)
valid_dataset = CustomDataset(data, partitions['valid'], tokenizer, feature_extractor)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_one_epoch(model, train_dataloader, optimizer)
    metrics = evaluate(model, valid_dataloader)

    wandb.log({
        "Epoch": epoch + 1,
        "Average Training Loss": train_loss,
        **metrics
    })

    # Save model checkpoint
    model.save_pretrained(f"vit_gpt2_finetuned_epoch_{epoch+1}")
    tokenizer.save_pretrained(f"vit_gpt2_finetuned_epoch_{epoch+1}")

# Finish WandB logging
wandb.finish()
