import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from Training.generators import generate_cross_states, generate_f2l_states


GENERATORS = {
    "cross": generate_cross_states,
    "f2l":   generate_f2l_states,
}


class ADIDataset(Dataset):
    def __init__(self, stage_name, stage, cube_class, n_samples=10000, max_scramble=20):
        if stage_name not in GENERATORS:
            raise ValueError(f"stage_name debe ser uno de: {list(GENERATORS.keys())}")

        generator    = GENERATORS[stage_name]
        self.samples = generator(cube_class, n_samples, max_scramble, stage.allowed_moves)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, target = self.samples[idx]
        return (
            torch.tensor(state,  dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )