import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import evaluate
import wandb
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    ViTModel,
    LlamaForCausalLM
)

# ---- Setup ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# W&B Init
wandb.init(project="C5_W3", entity="C3_MCV_LGVP", name="Test Evaluation")

# Paths
base_path = '../datasets/Food_Images/'
img_path = f'{base_path}Food_Images/'
cap_path = f'{base_path}Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'
partition_path = f'{base_path}food_partitions.npy'
llama_path = "llama/model_1_1B/llama_finetuned_epoch_10"

TEXT_MAX_LEN = 197

# ---- Load Data ----
data = pd.read_csv(cap_path)
partitions = np.load(partition_path, allow_pickle=True).item()

# Clean
data = data.dropna(subset=["Title"])
valid_images = list({os.path.splitext(f)[0] for f in os.listdir(img_path)})
data = data[data["Image_Name"].isin(valid_images)].reset_index(drop=True)
valid_indices = set(data.index)
partitions["test"] = [idx for idx in partitions["test"] if idx in valid_indices]

# ---- Metrics ----
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(llama_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- Feature extractor placeholder ----
feature_extractor = ViTModel.from_pretrained("models/vit_finetuned_epoch_10")

# ---- Dataset ----
class CustomDataset(Dataset):
    def __init__(self, data, partition, tokenizer, feature_extractor):
        self.data = data.loc[partition].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        base_transform = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]
        self.transform = transforms.Compose(base_transform)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
            caption = item['Title']

            tokenized = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                max_length=TEXT_MAX_LEN,
                truncation=True
            )
            labels = tokenized.input_ids[0]
            labels[labels == self.tokenizer.pad_token_id] = -100

            return image, labels, caption

        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            return None

# ---- Collate ----
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images, labels, captions = zip(*batch)
    return torch.stack(images), torch.stack(labels), list(captions)

# ---- Model ----
class VisionToLlamaModel(torch.nn.Module):
    def __init__(self, vision_model_name, llama_model_name):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.llama = LlamaForCausalLM.from_pretrained(llama_model_name, device_map="cuda")
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        self.llama = get_peft_model(self.llama, lora_config)

        vit_hidden = self.vision_encoder.config.hidden_size
        llama_hidden = self.llama.config.hidden_size
        self.projection = torch.nn.Linear(vit_hidden, llama_hidden)

    def forward(self, images, labels):
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            image_features = vision_outputs.last_hidden_state

        projected = self.projection(image_features)
        projected = projected.to(dtype=self.llama.dtype)
        outputs = self.llama(inputs_embeds=projected, labels=labels)
        return outputs

    def generate_from_image(self, images, max_new_tokens=50):
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            image_features = vision_outputs.last_hidden_state
            projected = self.projection(image_features)
            projected = projected.to(dtype=self.llama.dtype)

            outputs = self.llama.generate(
                inputs_embeds=projected,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.llama.config.eos_token_id,
                pad_token_id=self.llama.config.pad_token_id,
            )
        return outputs

# ---- Evaluation ----
def eval_one_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    all_predictions, all_references = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if batch is None:
                continue
            imgs, labels, captions = batch
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs, labels)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = model.generate_from_image(imgs)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            all_predictions.extend(preds)
            all_references.extend([[ref] for ref in captions])

    metrics = {
        "Test Loss": total_loss / len(dataloader),
        "BLEU-1": bleu.compute(predictions=all_predictions, references=all_references, max_order=1)["bleu"],
        "BLEU-2": bleu.compute(predictions=all_predictions, references=all_references, max_order=2)["bleu"],
        "ROUGE-L": rouge.compute(predictions=all_predictions, references=all_references)["rougeL"],
        "METEOR": meteor.compute(predictions=all_predictions, references=all_references)["meteor"]
    }

    wandb.log({f"Test/{k}": v for k, v in metrics.items()})
    return metrics

# ---- Load Model ----
model = VisionToLlamaModel(
    vision_model_name="models/vit_finetuned_epoch_10",
    llama_model_name=llama_path
).to(DEVICE)

# ---- Test Dataloader ----
test_dataset = CustomDataset(data, partitions["test"], tokenizer, feature_extractor)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# ---- Run Evaluation ----
print("\nRunning evaluation on test set...")
test_metrics = eval_one_epoch(model, test_dataloader)

print("\nTest Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

wandb.finish()
