import os
import numpy as np
import pandas as pd
import torch
# import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm
import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = './datasets/Food_Images/'
IMG_PATH = os.path.join(BASE_PATH, 'Food_Images/')
CAP_PATH = os.path.join(BASE_PATH, 'Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv')

PARTITION_FILES = [
    'food_partitions.npy',
    'food_partitions_plus_10pct_synthetic.npy',
    'food_partitions_plus_20pct_synthetic.npy',
    'food_partitions_plus_30pct_synthetic.npy',
    'food_partitions_plus_40pct_synthetic.npy',
    'food_partitions_plus_50pct_synthetic.npy',
    'food_partitions_only_synthetic.npy',
    'synthetic_food_partitions.npy'
]

EPOCHS = 20
BATCH_SIZE = 12
LEARNING_RATE = 5.33e-6
MODEL_NAME = 'nlpconnect/vit-gpt2-image-captioning'


# METRICS
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')


# DATASET CLASS
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
        img_path_full = os.path.join(IMG_PATH, img_name)

        if not os.path.exists(img_path_full):
            return None, None, None

        try:
            img = Image.open(img_path_full).convert('RGB')
            img_tensor = self.feature_extractor(images=img, return_tensors="pt").pixel_values[0]
            caption = item['Title']
            labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", max_length=64, truncation=True).input_ids[0]
            labels[labels == self.tokenizer.pad_token_id] = -100
            return img_tensor, labels, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None, None


def collate_fn(batch):
    imgs, labels, captions = zip(*[(img, label, caption) for img, label, caption in batch if img is not None])
    return torch.stack(imgs), torch.stack(labels), captions


# TRAINING
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
        # if step % step_log == 0:
        # wandb.log({"Training Loss": loss.item()})
    return total_loss / len(dataloader)


# EVALUATION
def evaluate_one_epoch(model, dataloader):
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
    # wandb.log(metrics)
    return metrics


# LOAD DATA
data = pd.read_csv(CAP_PATH)
dropped_indices = data[data['Title'].isna()].index
data = data.dropna(subset=['Title']).reset_index(drop=True)
valid_indices = set(data.index)

# MAIN LOOP
for partition_file in PARTITION_FILES:
    partition_path = os.path.join(BASE_PATH, partition_file)
    run_name = f"VIT+GPT2_{partition_file.replace('.npy', '')}"
    print(f"\nStarting config: {partition_file}")

    # Load partitions
    partitions = np.load(partition_path, allow_pickle=True).item()
    partitions['train'] = [idx for idx in partitions['train'] if idx not in dropped_indices and idx in valid_indices]
    partitions['valid'] = [idx for idx in partitions['valid'] if idx not in dropped_indices and idx in valid_indices]

    print(f"Training set size: {len(partitions['train'])}")
    print(f"Validation set size: {len(partitions['valid'])}")

    # Init W&B
    # wandb.init(project="C5_W5", entity="C3_MCV_LGVP", name=run_name)

    # Model
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(DEVICE)

    # (Don't freeze encoder – allow full fine-tuning)

    # Dataloaders
    train_dataset = CustomDataset(data, partitions['train'], tokenizer, feature_extractor)
    valid_dataset = CustomDataset(data, partitions['valid'], tokenizer, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer)
        metrics = evaluate_one_epoch(model, val_loader)
        # wandb.log({
        #    "Epoch": epoch + 1,
        #    "Average Training Loss": train_loss,
        #    **metrics
        # })

        if (epoch + 1) in [10, 20]:
            model_dir = f"model_output/{run_name}_epoch{epoch+1}"
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    # wandb.finish()
