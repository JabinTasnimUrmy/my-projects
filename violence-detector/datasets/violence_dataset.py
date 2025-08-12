

import os
import torch
from torch.utils.data import Dataset

class ViolenceTensorDataset(Dataset):
    """
    Loads pre-saved tensors for clips from the 'preprocessed_tensors' directory.
    This is much faster than loading individual images.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        classes = {'Fight': 1, 'NonFight': 0}

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"The directory '{root_dir}' does not exist. Please run the 'preprocess_to_tensors.py' script first.")

        for class_name, label in classes.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for sample_file in os.listdir(class_path):
                if sample_file.endswith('.pt'):
                    file_path = os.path.join(class_path, sample_file)
                    self.samples.append((file_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .pt tensor files found in {root_dir}! Please ensure you have run the 'preprocess_to_tensors.py' script and it generated files.")

        print(f"Found {len(self.samples)} tensor samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label = self.samples[idx]
        clip_tensor = torch.load(tensor_path)

        if self.transform:
            clip_tensor = self.transform(clip_tensor)

        return clip_tensor, torch.tensor(label, dtype=torch.float32)