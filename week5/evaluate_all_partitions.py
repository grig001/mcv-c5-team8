import os
import numpy as np
import pandas as pd
import torch
# import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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

MODEL_BASE_PATH = 'model_output'
MODEL_EPOCH = 10

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')


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
            return None

        try:
            img = Image.open(img_path_full).convert('RGB')
            img = self.feature_extractor(images=img, return_tensors="pt").pixel_values[0]
            caption = item['Title']
            return img, caption
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None


def collate_fn(batch):
    valid_samples = [(img, caption) for sample in batch if sample is not None for img, caption in [sample]]
    if not valid_samples:
        return torch.empty(0), []
    imgs, captions = zip(*valid_samples)
    return torch.stack(imgs), list(captions)


def evaluate_model(model, dataloader, tokenizer):
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

    return {
        "BLEU-1": bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"],
        "BLEU-2": bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"],
        "ROUGE-L": rouge.compute(predictions=all_predictions, references=all_references)["rougeL"],
        "METEOR": meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
    }


data = pd.read_csv(CAP_PATH)
dropped_indices = data[data['Title'].isna()].index
data = data.dropna(subset=['Title']).reset_index(drop=True)
valid_indices = set(data.index)

for partition_file in PARTITION_FILES:
    print(f"\nEvaluating model for partition: {partition_file}")

    # Load test set
    partition_path = os.path.join(BASE_PATH, partition_file)
    partitions = np.load(partition_path, allow_pickle=True).item()
    test_indices = [i for i in partitions['test'] if i not in dropped_indices and i in valid_indices]
    print(f"Test set size: {len(test_indices)}")

    # Load model
    model_dir_name = f"VIT+GPT2_{partition_file.replace('.npy', '')}_epoch{MODEL_EPOCH}"
    model_path = os.path.join(MODEL_BASE_PATH, model_dir_name)
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print("Model directory not found. Skipping.")
        continue

    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Dataset
    test_dataset = CustomDataset(data, test_indices, tokenizer, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # WandB init
    # wandb.init(project="C5_W5", entity="C3_MCV_LGVP", name=f"eval_{partition_file.replace('.npy','')}")

    # Evaluation
    metrics = evaluate_model(model, test_loader, tokenizer)
    # wandb.log(metrics)
    print(f"📊 Evaluation metrics for {partition_file}:\n", metrics)

    # wandb.finish()
