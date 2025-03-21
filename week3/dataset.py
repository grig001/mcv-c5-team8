from torch.utils.data import Dataset
import torch
from torchvision import transforms as v2
from PIL import Image
from constants import TEXT_MAX_LEN, CHAR2IDX, CHARS

# Path needs to be added in getitem


class FoodDataset(Dataset):
    def __init__(self, annotations):
        """Dataset for loading food images and their corresponding captions."""
        self.annotations = annotations
        self.max_len = TEXT_MAX_LEN

        # Define transformations
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Returns an image tensor, the raw caption, and the tokenized caption indices."""
        item = self.annotations.iloc[idx]

        # Load and process image
        img_path = "Path needs to be added" / item.Image_Name
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
    
        # Process caption
        caption = item.Title  # Original caption
        cap_tokens = [CHARS[0]] + list(caption) + [CHARS[1]]  # Add <SOS> and <EOS>

        # Pad the caption if necessary
        cap_tokens += [CHARS[2]] * (self.max_len - len(cap_tokens))

        # Convert characters to indices
        cap_idx = [CHAR2IDX[char] for char in cap_tokens]

        return img, caption, cap_idx
