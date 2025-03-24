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
from torchvision import models


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
valid_indices = set(data.index)

partitions['train'] = [idx for idx in partitions['train'] if idx in valid_indices]
partitions['valid'] = [idx for idx in partitions['valid'] if idx in valid_indices]
partitions['test'] = [idx for idx in partitions['test'] if idx in valid_indices]


def clean_text_before(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


data["Title"] = data["Title"].astype(str).apply(clean_text_before)

chars = list(set("".join(data["Title"])))
chars = ['<SOS>', '<EOS>', '<PAD>'] + sorted(chars)

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201


class Data(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.max_len = TEXT_MAX_LEN
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        real_idx = self.partition[idx]
        item = self.data.iloc[real_idx]

        img_name = item['Image_Name'] + '.jpg'
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)

        caption = item['Title']
        cap_list = list(caption)
        final_list = ['<SOS>'] + cap_list + ['<EOS>'] + ['<PAD>'] * (self.max_len - len(cap_list) - 2)
        cap_idx = [char2idx.get(c, char2idx['<PAD>']) for c in final_list]

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

            decoded_refs = [''.join([idx2char[idx.item()] for idx in caption if idx.item() != char2idx['<PAD>']])
                            for caption in captions]
            decoded_preds = [''.join([idx2char[idx.item()] for idx in pred if idx.item() != char2idx['<PAD>']])
                             for pred in predicted]

            if not decoded_preds:
                continue

            bleu1 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs], max_order=1)['bleu']
            bleu2 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs], max_order=2)['bleu']
            rouge_l = rouge.compute(predictions=decoded_preds, references=decoded_refs)['rougeL']
            meteor_s = meteor.compute(predictions=decoded_preds, references=decoded_refs)['meteor']

            bleu1_score += bleu1
            bleu2_score += bleu2
            rouge_score += rouge_l
            meteor_score += meteor_s

    avg_loss = total_loss / len(dataloader)
    return avg_loss, bleu1_score / len(dataloader), bleu2_score / len(dataloader), rouge_score / len(dataloader), meteor_score / len(dataloader)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg19 = nn.Sequential(vgg19, nn.AdaptiveAvgPool2d((1, 1)))
        self.lstm = nn.LSTM(512, 512, num_layers=3, dropout=0.3, bidirectional=True)
        self.proj = nn.Linear(1024, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)
        self.hidden_init = nn.Parameter(torch.zeros(6, 1, 512))
        self.cell_init = nn.Parameter(torch.zeros(6, 1, 512))
        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.vgg19(img)
        feat = feat.view(1, batch_size, 512)

        hidden = self.hidden_init.expand(-1, batch_size, -1).contiguous()
        cell = self.cell_init.expand(-1, batch_size, -1).contiguous()

        inp = self.embed(torch.tensor([char2idx['<SOS>']] * batch_size).to(DEVICE)).unsqueeze(0)
        outputs = []

        for _ in range(TEXT_MAX_LEN):
            out, (hidden, cell) = self.lstm(inp, (hidden, cell))
            out = self.layer_norm(out)
            logits = self.proj(out[-1])
            outputs.append(logits.unsqueeze(1))
            inp = self.embed(logits.argmax(dim=1)).unsqueeze(0)

        outputs = torch.cat(outputs, dim=1)
        return outputs.permute(0, 2, 1)


bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

crit = nn.CrossEntropyLoss()
data_test = Data(data, partitions['test'])
dataloader_test = DataLoader(data_test, batch_size=8, shuffle=True)

model_dir = "models/vgg19_lstm/"
available_models = [f for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pth")]
available_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

for model_file in available_models:
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file), map_location=DEVICE))
    model.eval()

    loss, bleu1, bleu2, rouge_l, meteor_s = eval_epoch(model, crit, dataloader_test)
    print(f"Model: {model_file}")
    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, ROUGE-L: {rouge_l:.4f}, METEOR: {meteor_s:.4f}")
    print("="*50)
