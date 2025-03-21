import pandas as pd
import numpy as np

# debuggin
# path = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C5_project/archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

def get_annotations(path):
    """Loads the annotations CSV and ensures image filenames have a '.jpg' extension."""
    df = pd.read_csv(path, index_col=0)
    df['Image_Name'] = df['Image_Name'].astype(str) + '.jpg'
    return df

def get_train_val_test_annotations_split(path, splits=(0.8, 0.1, 0.1)):
    """Splits the annotations dataset into train, validation, and test sets based on given proportions."""
    if not np.isclose(sum(splits), 1.0):
        raise ValueError("Splits must sum to 1.")

    df = get_annotations(path)
    n_samples = len(df)
    
    train_size = int(splits[0] * n_samples)
    val_size = int(splits[1] * n_samples)

    shuffled_indices = np.random.permutation(n_samples)
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    return {
        'train': df.iloc[train_indices].reset_index(drop=True),
        'val': df.iloc[val_indices].reset_index(drop=True),
        'test': df.iloc[test_indices].reset_index(drop=True)
    }


print(get_train_val_test_annotations_split(path))
