import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BaseAgent(nn.Module):
    def __init__(self, input_size, output_size):
        self.model = NeuralNetwork(input_size, output_size)

    

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # (no activation if regression, use softmax if classification)
        return x
    
    def train(self, data_loader):
        self.model.train()  # Set model to training mode
        for batch_data, batch_labels in data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(network, X, y):
        with torch.no_grad():
            output = network(X)
            predictions = torch.argmax(output, dim=1)
            return (predictions == y).float().mean().item()