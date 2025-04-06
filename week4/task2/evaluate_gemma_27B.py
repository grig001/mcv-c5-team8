import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import evaluate
import wandb

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.bfloat16)  # Gemma uses bfloat16

# Init W&B
wandb.init(project="Gemma3_Image_Captioning", entity="C3_MCV_LGVP")

# Metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

# Paths
base_path = "../datasets/Food_Images/"
img_path = os.path.join(base_path, "Food_Images")
csv_path = os.path.join(base_path, "Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv")
partitions = np.load(os.path.join(base_path, "food_partitions.npy"), allow_pickle=True).item()

# Load Data
data = pd.read_csv(csv_path)
dropped_indices = data[data['Title'].isna()].index
partitions["test"] = [idx for idx in partitions["test"] if idx not in dropped_indices]
data = data.dropna(subset=["Title"]).reset_index(drop=True)
valid_indices = set(data.index)
partitions["test"] = [idx for idx in partitions["test"] if idx in valid_indices]

# Dataset
class FoodDataset(Dataset):
    def __init__(self, data, partition):
        self.data = data.loc[partition].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = f"{row['Image_Name']}.jpg"
        full_path = os.path.join(img_path, img_name)

        try:
            image = Image.open(full_path).convert("RGB")
            caption = row["Title"]
            return image, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    images, captions = zip(*batch)
    return list(images), list(captions)

# Load model and processor
model_id = "google/gemma-3-12b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

# Evaluation
def evaluate_model(model, processor, dataloader):
    model.eval()
    predictions, references = [], []

    for images, gold_captions in tqdm(dataloader, desc="Evaluating"):
        if not images:
            continue

        batch_preds = []
        for img in images:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this food image in detail."}
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                gen_out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generated = gen_out[0][inputs["input_ids"].shape[-1]:]
                decoded = processor.decode(generated, skip_special_tokens=True)
                batch_preds.append(decoded.strip())

        predictions.extend(batch_preds)
        references.extend([[ref] for ref in gold_captions])

    metrics = {
        "BLEU-1": bleu.compute(predictions=predictions, references=references, max_order=1)["bleu"],
        "BLEU-2": bleu.compute(predictions=predictions, references=references, max_order=2)["bleu"],
        "ROUGE-L": rouge.compute(predictions=predictions, references=references)["rougeL"],
        "METEOR": meteor.compute(predictions=predictions, references=references)["meteor"]
    }
    return metrics

# DataLoader
test_dataset = FoodDataset(data, partitions["test"])
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Run Evaluation
metrics = evaluate_model(model, processor, test_dataloader)
print("Evaluation Metrics:", metrics)
wandb.log(metrics)

# Clean up
wandb.finish()
torch.cuda.empty_cache()
