import os
import torch
import evaluate

import numpy as np
import pandas as pd
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from transformers import (
                            AutoTokenizer, ViTModel, LlamaForCausalLM, get_scheduler
                        )

import wandb

DEVICE = torch.device("cuda")

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)


wandb.init(project="C5_W3", entity="C3_MCV_LGVP")

base_path = 'D:/mcv-c5-team8-1/week3/datasets/Food_Images/'
# base_path = '../datasets/Food_Images/'

TEXT_MAX_LEN = 197

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
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

class CustomDataset(Dataset):
    def __init__(self, data, partition, tokenizer, feature_extractor, augmentation=False):
        self.data = data.loc[partition].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.augmentation = augmentation

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Augmentation
        base_transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]
        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.95, 1.0), ratio=(0.98, 1.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
                *base_transform
            ])
        else:
            self.transform = transforms.Compose(base_transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_name = item['Image_Name'] + '.jpg'
        img_path_full = os.path.join(img_path, img_name)

        if not os.path.exists(img_path_full):
            print(f"Missing image: {img_path_full}")
            return None

        try:
            image = Image.open(img_path_full).convert('RGB')
            image = self.transform(image)

            # Tokenize the caption
            caption = item['Title']
            tokenized = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                max_length=TEXT_MAX_LEN,
                truncation=True
            )

            labels = tokenized.input_ids[0]
            labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding for loss

            return image, labels

        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            return None


class VisionToLlamaModel(nn.Module):
    def __init__(self, vision_model_name, llama_model_name):
        super().__init__()

        # Frozen vision encoder
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # LLaMA decoder
        self.llama = LlamaForCausalLM.from_pretrained(llama_model_name, device_map="cuda")
        self.llama = get_peft_model(self.llama, lora_config)

        # Project ViT features to LLaMA hidden size
        vit_hidden = self.vision_encoder.config.hidden_size
        llama_hidden = self.llama.config.hidden_size
        self.projection = nn.Linear(vit_hidden, llama_hidden)


    def forward(self, images, labels):
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            image_features = vision_outputs.last_hidden_state  # [B, N, D_vit]

        # print("image features dim:", image_features.shape)
        projected = self.projection(image_features)  # [B, N, D_llama]

        # Forward through LLaMA using embeddings (no input_ids)
        projected = projected.to(dtype=self.llama.dtype)
        # print(projected.shape)
        # print(labels.shape)
        # embedding_size = self.llama.get_input_embeddings().embedding_dim
        # print(projected.shape)
        
        # expected_dim = self.llama.get_input_embeddings().embedding_dim
        # print("Expected embedding dim:", expected_dim)

        # Inspect actual projected shape
        # print("Projected shape:", projected.shape)

        # Sanity check
        # print(projected.shape[-1] == expected_dim)

        outputs = self.llama(inputs_embeds=projected, labels=labels)
        
        # print(outputs)

        return outputs

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionToLlamaModel(
    vision_model_name="D:/mcv-c5-team8-1/vit/model_0/vit_finetuned_epoch_10",  # or your own pretrained ViT
    llama_model_name="meta-llama/Llama-3.2-1B"              # or the model you downloaded
).to(device)


# Metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

# pre_vit = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTModel.from_pretrained("D:/mcv-c5-team8-1/vit/model_0/vit_finetuned_epoch_10")

train_dataset = CustomDataset(
    data=data,                      # your DataFrame
    partition=partitions['train'],      # train split
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,  # from transformers ViTFeatureExtractor
    augmentation=False
)

valid_dataset = CustomDataset(
    data=data,                      # your DataFrame
    partition=partitions['valid'],      # train split
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,  # from transformers ViTFeatureExtractor
    augmentation=False
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


# print(100 * '_')
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

'________________________________________________________________________________________________________________________'

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # print("image shape: ", images.shape)
        # print("labels shape: ", labels.shape)

        outputs = model(images, labels)

        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def greedy_decode(model, images, tokenizer, max_length=197, temperature=1.0):
    # Extract the features from the vision encoder (ViT)
    # vision_features = model.vision_encoder(images)  # This will give you features from ViT
    
    # Start with a token (usually <s> token for start, assuming token ID = 1)
    input_ids = torch.tensor([[1]]).to(images.device)  # Use the starting token
    
    generated_ids = input_ids  # Initialize generated_ids with the starting token

    eos_token_id = tokenizer.eos_token_id  # Get the EOS token ID from the tokenizer

    for _ in range(max_length):

        outputs = model.llama.base_model.model(generated_ids)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token in the sequence
        logits = logits / temperature  # Apply temperature scaling for diversity

        # Greedily pick the next token
        predicted_id = logits.argmax(dim=-1, keepdim=True)

        # Append the predicted token to the generated sequence
        generated_ids = torch.cat((generated_ids, predicted_id), dim=-1)

        # Check if EOS token is predicted and stop generation if EOS is reached
        if predicted_id.item() == eos_token_id:  # Using .item() to get the scalar value
            break

    generated_ids = generated_ids.squeeze().cpu().numpy().astype(int)  # Ensure it's a list of IDs

    return generated_ids


def model_evaluation(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Decoder Evaluation"):
            if batch is None:
                continue

            images, labels = batch  # captions = list of reference texts
            images, labels = images.to(device), labels.to(device)

            # Forward pass to compute loss
            outputs = model(images, labels)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = greedy_decode(model, images, tokenizer, max_length=197, temperature=1.0)

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = [[ref] for ref in labels]  # wrap each reference in a list for BLEU

            all_predictions.extend(preds)
            all_references.extend(refs)

    # Compute metrics
    metrics = {
        "Validation Loss": total_loss / len(dataloader),
        "BLEU-1": bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"],
        "BLEU-2": bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"],
        "ROUGE-L": rouge.compute(predictions=all_predictions, references=all_references)["rougeL"],
        "METEOR": meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
    }

    wandb.log(metrics)
    return metrics


def train(model, train_dataloader, valid_dataloader, optimizer, scheduler, EPOCHS = 4):
    
    for epoch in range(EPOCHS):
        # Train for one epoch
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, device)

        # Evaluation & Logging
        metrics = model_evaluation(model, valid_dataloader)

        # Log metrics to WandB
        wandb.log({
                "Epoch": epoch + 1,
                "Average Training Loss": avg_train_loss,
                **metrics
            })

        # Create the folder if it doesn't exist
        os.makedirs("llama/model_1", exist_ok=True)

        # Save model checkpoint
        model.save_pretrained(f"llama/model_1/llama_finetuned_epoch_{epoch+1}")
        tokenizer.save_pretrained(f"llama/model_1/llama_finetuned_epoch_{epoch+1}")
            
        # Scheduler step (for learning rate adjustment)
        scheduler.step()


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 4)

train(model, train_dataloader, valid_dataloader, optimizer, lr_scheduler)

wandb.finish()
