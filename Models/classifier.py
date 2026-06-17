import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512, num_layers=4):
        super().__init__()
        
        layers = []
        in_size = input_size

        for _ in range(num_layers):
            linear = nn.Linear(in_size, hidden_size)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ])
            in_size = hidden_size

        out = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)