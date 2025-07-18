import json
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class PklDatasetWithLabel(Dataset):
    def __init__(self, json_path, label_list,random_crop_chance = 0.5,
            random_flip_chance = 0.5,
            random_rotate_chance = 0.5,):
        """
        Args:
            json_path (str): Path to the JSON file storing {label: [list of pkl paths]}.
            label_list (list): List of all possible labels (for one-hot encoding).
        """
        self.label_list = label_list
        self.label_to_index = {label: idx for idx, label in enumerate(label_list)}
        self.random_crop_chance = random_crop_chance
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance

        # Load dict from JSON
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        # Flatten and index
        self.samples = []
        for label, pkl_paths in self.data_dict.items():
            for path in pkl_paths:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load .pkl file
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Convert label to one-hot tensor
        label_idx = self.label_to_index[label]
        label_tensor = torch.zeros(len(self.label_list) + 1, dtype=torch.float32)
        label_tensor[label_idx] = 1.0

        return data, label_tensor
