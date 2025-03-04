import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BaseAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, output_size)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)             # (no activation if regression, use softmax if classification)
        return x

    # Evaluation function (accuracy)
    def evaluate(network, X, y):
        with torch.no_grad():
            output = network(X)
            predictions = torch.argmax(output, dim=1)
            return (predictions == y).float().mean().item()