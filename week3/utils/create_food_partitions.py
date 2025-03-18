import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
annotations_file = 'datasets/Food_Images/Food_Ingredients_and_Recipe_Dataset_with_Image_Name_Mapping.csv'
output_file = 'datasets/Food_Images/food_partitions.npy'

# Load the dataset
data = pd.read_csv(annotations_file)

# Generate indices
indices = np.arange(len(data))  # Indices for all rows in the dataset

# Split dataset (80% train, 10% validation, 10% test)
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)  # 80% train, 20% temp
val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)  # 10% validation, 10% test

# Store partitions
partitions = {
    'train': train_indices.tolist(),
    'valid': val_indices.tolist(),
    'test': test_indices.tolist()
}

# Save as .npy file
np.save(output_file, partitions)

print(f"✅ Correct dataset split complete! Partitions saved in {output_file}")
print(f"📂 Train: {len(train_indices)} indices")
print(f"📂 Validation: {len(val_indices)} indices")
print(f"📂 Test: {len(test_indices)} indices")
