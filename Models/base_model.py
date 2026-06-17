import torch
import torch.nn as nn

class StageModel(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        layers = []
        in_size = input_size
        
        layers.extend([
            nn.Linear(in_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        ])
        out = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)