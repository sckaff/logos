import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input, output):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Hidden activation
        x = self.fc2(x)  # Output logits
        return x


# Evaluation function (accuracy)
def evaluate(network, X, y):
    with torch.no_grad():
        output = network(X)
        predictions = torch.argmax(output, dim=1)
        return (predictions == y).float().mean().item()
