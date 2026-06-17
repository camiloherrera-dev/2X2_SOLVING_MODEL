import torch
from torch.utils.data import Dataset

class ClassifierDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, label = self.samples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )