import torch
from torch.utils.data import Dataset
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, path):
        # Load the dataset from the provided path
        self.data = np.load(path)
        
    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single instance from the dataset
        return torch.from_numpy(self.data[idx]).float()