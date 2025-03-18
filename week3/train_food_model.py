import torch
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from transformers import ResNetModel
from torch.utils.data import DataLoader
import torch.optim as optim

print("Starting script...")

DEVICE = "cuda"

# Paths
IMG_PATH = 'datasets/Food_Images/Food_Images/'
ANNOTATIONS_PATH = 'datasets/Food_Images/Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'
PARTITIONS_PATH = 'datasets/Food_Images/food_partitions.npy'

# Load data
data = pd.read_csv(ANNOTATIONS_PATH)
partitions = np.load(PARTITIONS_PATH, allow_pickle=True).item()

# Character encoding
chars = ['<SOS>', '<EOS>', '<PAD>'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?")
NUM_CHAR = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

TEXT_MAX_LEN = 100  # Max length of recipe title


class FoodDataset(Dataset):
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
        real_idx = self.partition[idx]  # Row index in dataset
        item = self.data.iloc[real_idx]  # Get row

        # Load and process image
        img_name = item.Image_Name
        img = Image.open(f"{IMG_PATH}{img_name}.jpg").convert("RGB")
        img = self.img_proc(img)

        # Process recipe title
        title = item.Title
        title_list = list(title)

        final_list = ["<SOS>"] + title_list + ["<EOS>"]
        gap = self.max_len - len(final_list)
        final_list.extend(["<PAD>"] * gap)

        title_idx = [char2idx.get(c, char2idx[" "]) for c in final_list]

        return img, torch.tensor(title_idx, dtype=torch.long)


class FoodCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-18").to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output.unsqueeze(0)  # (1, batch, 512)
        start = torch.tensor(char2idx["<SOS>"]).to(DEVICE)
        start_embed = self.embed(start).repeat(batch_size, 1).unsqueeze(0)  # (1, batch, 512)

        inp = start_embed
        hidden = feat
        for _ in range(TEXT_MAX_LEN - 1):  # Remove <SOS>
            out, hidden = self.gru(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0)

        res = inp.permute(1, 0, 2)  # (batch, seq, 512)
        res = self.proj(res).permute(0, 2, 1)  # (batch, vocab_size, seq)

        return res


def train(EPOCHS, batch_size=16):
    # Load dataset and create DataLoaders
    train_data = FoodDataset(data, partitions["train"])
    valid_data = FoodDataset(data, partitions["valid"])
    test_data = FoodDataset(data, partitions["test"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model
    model = FoodCaptioningModel().to(DEVICE)
    model.train()

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0

        for img, title in train_loader:
            img, title = img.to(DEVICE), title.to(DEVICE)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, title)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    EPOCHS = 10  # Set the number of epochs
    BATCH_SIZE = 16  # Define batch size

    print("Starting training...")
    train(EPOCHS, batch_size=BATCH_SIZE)
    print("Training finished!")
