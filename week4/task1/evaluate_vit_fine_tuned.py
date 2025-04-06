import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
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
partitions['test'] = [idx for idx in partitions['test'] if idx not in dropped_indices]
data = data.dropna(subset=['Title'])
data = data.reset_index(drop=True)
valid_indices = set(data.index)
partitions['test'] = [idx for idx in partitions['test'] if idx in valid_indices]


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
            return None  # Instead of returning (None, None, None)

        try:
            img = Image.open(img_path_full).convert('RGB')
            img = self.feature_extractor(images=img, return_tensors="pt").pixel_values[0]
            caption = item['Title']
            return img, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None  # Instead of returning (None, None)


def collate_fn(batch):
    valid_samples = [(img, caption) for sample in batch if sample is not None for img, caption in [sample]]
    if len(valid_samples) == 0:
        return torch.empty(0), []

    imgs, captions = zip(*valid_samples)
    imgs = torch.stack(imgs)
    return imgs, captions


# Load Fine-Tuned Model
model_name = "model_output/vit_finetuned_epoch_10"  # Change this to your desired model checkpoint
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(DEVICE)
model.eval()

# Prepare Test DataLoader
test_dataset = CustomDataset(data, partitions['test'], tokenizer, feature_extractor)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for imgs, captions in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            generated_ids = model.generate(imgs)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_predictions.extend(preds)
            all_references.extend([[ref] for ref in captions])

    metrics = {
        "BLEU-1": bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"],
        "BLEU-2": bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"],
        "ROUGE-L": rouge.compute(predictions=all_predictions, references=all_references)["rougeL"],
        "METEOR": meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
    }

    # Log results to wandb
    wandb.log(metrics)
    return metrics


# Evaluate the Model
metrics = evaluate_model(model, test_dataloader)
print("Evaluation Metrics:", metrics)

# Finish WandB logging
wandb.finish()
