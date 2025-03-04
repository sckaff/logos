import torch
import torch.nn as nn
import torch.optim as optim
from faker import Faker

faker = Faker()

class BaseAgent(nn.Module):
    def __init__(self, input_size, output_size, name=None):
        super().__init__()
        self.model = NeuralNetwork(input_size, output_size)
        if name is None:
            self.name = faker.name()
        else:
            self.name = name
        self.train_loader = None
        self.test_loader = None
        self.current_accuracy = 0

    def load_train_data(self, dataloader):
        self.train_loader = dataloader

    def load_test_data(self, dataloader):
        self.test_loader = dataloader

    def get_name(self):
        return self.name

    def forward(self, x):
        return self.model(x)
    
    def train(self, epochs=1, lr=0.001):
        """Trains the agent's neural network."""
        if self.train_loader is None:
            raise ValueError("Train loader not loaded. Call load_train_data first.")

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"Agent {self.name} Epoch {epoch + 1}/{epochs}")
            self.model.train_net(self.train_loader, optimizer)

        self.test()

    def test(self):
            # Test the agent after each epoch
            test_accuracy = 0.0
            with torch.no_grad():
                for X, y in self.test_loader:
                    test_accuracy += self.model.test_net(X, y)
            test_accuracy /= len(self.test_loader)
            self.current_accuracy = test_accuracy

    def accuracy(self):
        return self.current_accuracy


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1) #Flatten the input.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self, data_loader, optimizer):
        self.train()
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = self.forward(batch_data)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    def test_net(self, X, y):
        self.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1)
            output = self.forward(X)
            predictions = torch.argmax(output, dim=1)
            return (predictions == y).float().mean().item()