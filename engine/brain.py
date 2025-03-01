import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)

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


# Mutation function
def mutate(network, mutation_rate=0.05):
    child = NeuralNetwork()
    child.load_state_dict(network.state_dict())  # Copy weights
    
    for param in child.parameters():
        if len(param.shape) > 1:  # Only mutate weights, not biases
            param.data += torch.randn_like(param) * mutation_rate
            
    return child