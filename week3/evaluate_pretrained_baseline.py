#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import pandas as pd
import evaluate
from torch.utils.data import DataLoader
import os
import unicodedata
from transformers import ResNetModel


# DEVICE = 'cuda'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = 'datasets/Food_Images/'
img_path = f'{base_path}Food_Images/'
cap_path = f'{base_path}Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'

data = pd.read_csv(cap_path)
partitions = np.load("datasets/Food_Images/food_partitions.npy", allow_pickle=True).item()

dropped_indices = data[data["Title"].isna()].index
partitions['train'] = [idx for idx in partitions['train'] if idx not in dropped_indices]
data = data.dropna(subset=["Title"])

image_folder = "datasets/Food_Images/Food_Images"
valid_images = list({os.path.splitext(f)[0] for f in os.listdir(image_folder)})

data = data[data["Image_Name"].isin(valid_images)]

# Reset index after filtering
data = data.reset_index(drop=True)
valid_indices = set(data.index)  # These are the indices that remain after filtering

partitions['train'] = [idx for idx in partitions['train'] if idx in valid_indices]
partitions['valid'] = [idx for idx in partitions['valid'] if idx in valid_indices]
partitions['test'] = [idx for idx in partitions['test'] if idx in valid_indices]


# Normalize and remove unwanted characters
def clean_text_before(text):
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode
    text = text.encode("ascii", "ignore").decode("ascii")  # Remove non-ASCII chars
    return text


# Apply cleaning to the Title column
data["Title"] = data["Title"].astype(str).apply(clean_text_before)

# Extract unique characters
chars = list(set("".join(data["Title"])))

# Ensure special tokens are first
chars = ['<SOS>', '<EOS>', '<PAD>'] + sorted(chars)

# chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201


class Data(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.num_captions = 5
        self.max_len = TEXT_MAX_LEN
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        real_idx = self.partition[idx]  # Row index in dataset
        item = self.data.iloc[real_idx]  # Get row

        img_name = item.Image_Name + '.jpg'
        # print(img_name)
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)

        caption = item["Title"]
        cap_list = list(caption)

        final_list = [chars[0]]
        final_list.extend(cap_list)
        final_list.extend([chars[1]])
        gap = self.max_len - len(final_list)
        final_list.extend([chars[2]]*gap)

        missing_chars = [c for c in final_list if c not in char2idx]
        if missing_chars:
            print(f"Missing characters: {set(missing_chars)}")

        for char in missing_chars:
            if char not in char2idx:
                char2idx[char] = len(char2idx)  # Assign a new index

        cap_idx = [char2idx[i] for i in final_list]

        # return img, cap_idx
        return img, torch.tensor(cap_idx, dtype=torch.long)


def eval_epoch(model, crit, dataloader):
    total_loss = 0.0

    bleu1_score = 0
    bleu2_score = 0
    rouge_score = 0
    meteor_score = 0

    with torch.no_grad():
        for imgs, captions in dataloader:
            imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)
            outputs = model(imgs)
            loss = crit(outputs, captions)
            total_loss += loss.item()

            _, predicted = outputs.max(1)

            decoded_refs = [clean_text(decode_caption(caption.cpu().numpy(), chars)) for caption in captions]
            decoded_preds = [clean_text(decode_caption(pred.cpu().numpy(), chars)) for pred in predicted]

            # print(f"Ref: {decoded_refs}")
            # print(f"Pred: {decoded_preds}")
            if is_empty_prediction(decoded_preds):
                continue

            bleu1 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs], max_order=1)
            bleu2 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs], max_order=2)
            res_r = rouge.compute(predictions=decoded_preds, references=decoded_refs)
            res_m = meteor.compute(predictions=decoded_preds, references=decoded_refs)

            # Accumulate scores
            bleu1_score += bleu1["bleu"]
            bleu2_score += bleu2["bleu"]
            rouge_score += res_r["rougeL"]
            meteor_score += res_m["meteor"]

    avg_loss = total_loss / len(dataloader)
    bleu1_score /= len(dataloader)
    bleu2_score /= len(dataloader)
    rouge_score /= len(dataloader)
    meteor_score /= len(dataloader)

    return avg_loss, bleu1_score, bleu2_score, rouge_score, meteor_score


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)
        self.num_classes = NUM_CHAR

    def forward(self, img, captions=None, teacher_forcing_ratio=0.5):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, batch, 512)

        start_token = torch.full((batch_size,), char2idx['<SOS>'], dtype=torch.long, device=DEVICE)
        start_embed = self.embed(start_token).unsqueeze(0)  # (1, batch, 512)

        hidden = feat
        inp = start_embed
        outputs = []

        for t in range(TEXT_MAX_LEN):  # Exclude <SOS>
            out, hidden = self.gru(inp, hidden)
            logits = self.proj(out[-1])  # (batch, NUM_CHAR)

            outputs.append(logits.unsqueeze(1))  # Store timestep output

            # Decide whether to use teacher forcing
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                inp = self.embed(captions[:, t]).unsqueeze(0)  # Use ground truth token
            else:
                pred = logits.argmax(dim=1)
                inp = self.embed(pred).unsqueeze(0)  # Use model prediction

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, NUM_CHAR)
        return outputs.permute(0, 2, 1)  # (batch, NUM_CHAR, seq_len)


# Metrics
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')


def decode_caption(indices, vocab):
    return ''.join([vocab[idx] if idx < len(vocab) else '<UNK>' for idx in indices if idx not in [0]])


def clean_text(text):
    """Removes padding and special tokens, then strips whitespace."""
    return text.replace("<PAD>", "").replace("<EOS>", "").strip()


def is_empty_prediction(pred_list):
    """Checks if any cleaned prediction is empty."""
    return any(len(clean_text(pred)) == 0 for pred in pred_list)


# Evaluation
crit = nn.CrossEntropyLoss()

batch_size = 8
data_test = Data(data, partitions['test'])
dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=0)

model = Model().to(DEVICE)
model.eval()

avg_loss, bleu1_score, bleu2_score, rouge_score, meteor_score = eval_epoch(model, crit, dataloader_test)
print("Pretrained Baseline Model - Metrics: \n")
print(f"BLEU-1: {bleu1_score:.4f}")
print(f"BLEU-2: {bleu2_score:.4f}")
print(f"ROUGE-L: {rouge_score:.4f}")
print(f"METEOR: {meteor_score:.4f}")
