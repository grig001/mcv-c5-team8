import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
import evaluate
from torch import nn
import unicodedata


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load test partition from npy file
partitions = np.load('datasets/Food_Images/food_partitions.npy', allow_pickle=True).item()

# Load dataset
base_path = 'datasets/'
img_path = f'{base_path}Food_Images/Food_Images/'
cap_path = f'{base_path}Food_Images/Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'

data = pd.read_csv(cap_path)
data = data.dropna(subset=['Title'])

# Load valid image names
valid_images = list({f[:-4] for f in os.listdir(img_path)})
data = data[data['Image_Name'].isin(valid_images)]
data = data.reset_index(drop=True)

# Filter valid indices
valid_indices = set(data.index)
partitions['test'] = [idx for idx in partitions['test'] if idx in valid_indices]


# Character Mapping
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


data["Title"] = data["Title"].astype(str).apply(clean_text)
chars = list(set("".join(data["Title"])))
chars = ['<SOS>', '<EOS>', '<PAD>'] + sorted(chars)
char2idx = {v: k for k, v in enumerate(chars)}
idx2char = {k: v for k, v in enumerate(chars)}
NUM_CHAR = len(chars)
TEXT_MAX_LEN = 201


# Define Dataset Class
class Data(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.max_len = TEXT_MAX_LEN
        self.img_proc = nn.Sequential(
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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features  # Load pretrained VGG19
        self.vgg19 = nn.Sequential(vgg19, nn.AdaptiveAvgPool2d((1, 1)))

        self.lstm = nn.LSTM(512, 512, num_layers=3, dropout=0.3, bidirectional=True)
        self.proj = nn.Linear(1024, NUM_CHAR)  # 1024 because of the bidirectional LSTM
        self.embed = nn.Embedding(NUM_CHAR, 512)

        self.hidden_init = nn.Parameter(torch.zeros(6, 1, 512))  # (num_layers * 2, batch, hidden_size)
        self.cell_init = nn.Parameter(torch.zeros(6, 1, 512))

        self.layer_norm = nn.LayerNorm(1024)  # Normalize LSTM outputs

    def forward(self, img):
        batch_size = img.shape[0]

        # Feature extraction using VGG19
        feat = self.vgg19(img)
        feat = feat.view(1, batch_size, 512)  # Reshape to (1, batch_size, 512)

        # Initialize LSTM hidden states
        hidden = self.hidden_init.expand(-1, batch_size, -1).contiguous()
        cell = self.cell_init.expand(-1, batch_size, -1).contiguous()

        # Embedding for <SOS> token
        inp = self.embed(torch.tensor([char2idx['<SOS>']] * batch_size).to(DEVICE)).unsqueeze(0)

        outputs = []
        for _ in range(TEXT_MAX_LEN):
            out, (hidden, cell) = self.lstm(inp, (hidden, cell))
            out = self.layer_norm(out)  # Apply layer normalization
            logits = self.proj(out[-1])  # (batch, NUM_CHAR)
            outputs.append(logits.unsqueeze(1))
            inp = self.embed(logits.argmax(dim=1)).unsqueeze(0)

        outputs = torch.cat(outputs, dim=1)
        return outputs.permute(0, 2, 1)  # (batch, NUM_CHAR, seq_len)


# Load model
model = Model().to(DEVICE)
model.eval()


# Evaluation
test_dataset = Data(data, partitions['test'])
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

total_bleu1, total_bleu2, total_rouge, total_meteor = 0, 0, 0, 0

for imgs, captions in test_loader:
    imgs, captions = imgs.to(DEVICE), captions.to(DEVICE)
    outputs = model(imgs)
    preds = torch.argmax(outputs, dim=1)

    decoded_preds = [''.join([idx2char[idx.item()] for idx in pred if idx.item() != char2idx['<PAD>']]) for pred in preds]
    decoded_refs = [''.join([idx2char[idx.item()] for idx in caption if idx.item() != char2idx['<PAD>']]) for caption in captions]

    bleu1 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs], max_order=1)['bleu']
    bleu2 = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs], max_order=2)['bleu']
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_refs)['rougeL']
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)['meteor']

    total_bleu1 += bleu1
    total_bleu2 += bleu2
    total_rouge += rouge_score
    total_meteor += meteor_score


num_batches = len(test_loader)
print(f'BLEU-1: {total_bleu1 / num_batches:.4f}')
print(f'BLEU-2: {total_bleu2 / num_batches:.4f}')
print(f'ROUGE-L: {total_rouge / num_batches:.4f}')
print(f'METEOR: {total_meteor / num_batches:.4f}')
