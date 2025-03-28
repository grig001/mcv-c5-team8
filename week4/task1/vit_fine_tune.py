import os
import torch
import wandb
import optuna
import evaluate

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AdamW, get_scheduler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.finish()


wandb.init(project="C5_W3", entity="C3_MCV_LGVP")


base_path = '../datasets/Food_Images/'
img_path = f'{base_path}Food_Images/'
cap_path = f'{base_path}Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'
data = pd.read_csv(cap_path)
partitions = np.load(f'{base_path}food_partitions.npy', allow_pickle=True).item()

# Clean Data
data = data.dropna(subset=["Title"])

valid_images = list({os.path.splitext(f)[0] for f in os.listdir(img_path)})
data = data[data["Image_Name"].isin(valid_images)]
data = data.reset_index(drop=True)

valid_indices = set(data.index)

partitions['train'] = [idx for idx in partitions['train'] if idx in valid_indices]
partitions['valid'] = [idx for idx in partitions['valid'] if idx in valid_indices]

def collate_fn(batch):
    batch = [(img, label, caption) for img, label, caption in batch if img is not None]   
    imgs, labels, captions = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  
    labels = torch.stack(labels, dim=0)
    return imgs, labels, captions

class ModelLoader:
    def __init__(self, vit_model_name="google/vit-base-patch16-224", gpt2_model_name="gpt2"):
        self.vit_model_name = vit_model_name
        self.gpt2_model_name = gpt2_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the VisionEncoderDecoderModel and tokenizer
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        
    def load_model(self):
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model.to(self.device)

        # Freeze GPT-2 (text generation) part of the model
        for param in model.decoder.parameters():  # Freeze GPT-2 part (decoder)
            param.requires_grad = False
        
        for param in model.encoder.parameters():
            param.requires_grad = True
        
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        return tokenizer


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, partition, tokenizer, feature_extractor, augmentation=False):
        self.data = data.loc[partition].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.augmentation = augmentation

        # Define augmentation pipeline
        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.95, 1.0), ratio=(0.98, 1.02)),  # Slight crop, keeps structure intact
                transforms.RandomHorizontalFlip(p=0.5),  # Safe augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Light color variations
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),  # Mild blur for robustness
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization for ViT
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_name = item['Image_Name'] + '.jpg'
        img_path_full = f'{img_path}{img_name}'

        if not os.path.exists(img_path_full):
            print(f"Image not found: {img_path_full}")
            return None, None, None

        try:
            img = Image.open(img_path_full).convert('RGB')

            if self.transform:
                img = self.transform(img)

            img = self.feature_extractor(images=img, return_tensors="pt").pixel_values[0]

            caption = item['Title']
            labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", max_length=128, truncation=True).input_ids[0]
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding

            return img, labels, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None, None



# Load models
model_loader = ModelLoader()
vit_model = model_loader.load_model()
tokenizer = model_loader.load_tokenizer()

feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

# Data Loaders
train_dataset = CustomDataset(data, partitions['train'], tokenizer, feature_extractor, augmentation=False)
valid_dataset = CustomDataset(data, partitions['valid'], tokenizer, feature_extractor, augmentation=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# Training Function
def train_one_epoch(model, dataloader, optimizer, step_log=500):
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



def train(model, train_dataloader, valid_dataloader, optimizer, scheduler, EPOCHS = 5):
    
    for epoch in range(EPOCHS):
        # Train for one epoch
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer)

        # Evaluation & Logging
        metrics = evaluate(model, valid_dataloader)

        # Log metrics to WandB
        wandb.log({
                "Epoch": epoch + 1,
                "Average Training Loss": avg_train_loss,
                **metrics
            })

        os.makedirs("vit/model_4", exist_ok=True)

        # Save model checkpoint
        model.save_pretrained(f"vit/model_4/vit_finetuned_epoch_{epoch+1}")
        tokenizer.save_pretrained(f"vit/model_4/vit_finetuned_epoch_{epoch+1}")
            
        scheduler.step()
        

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_predictions, all_references = [], []
    
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


# Make sure lr_scheduler is defined earlier as you did before:
optimizer = AdamW(vit_model.parameters(), lr=5e-5, weight_decay=1e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 5)

train(vit_model, train_dataloader, valid_dataloader, optimizer, lr_scheduler)

wandb.finish()
