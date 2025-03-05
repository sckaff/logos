import torch
import torch.nn as nn
import torch.optim as optim
from faker import Faker
from utils.utils import format_float

faker = Faker()

class BaseAgent(nn.Module):
    """
    Base class for agents with a neural network model.
    """
    def __init__(self, input_size, output_size, name=None):
        """
        Initializes the BaseAgent.

        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            name (str, optional): Name of the agent. If None, a random name is generated.
        """
        super().__init__()
        self.model = NeuralNetwork(input_size, output_size)
        if name is None:
            self.name = faker.name()
        else:
            self.name = name
        self.train_loader = None
        self.test_loader = None
        self._accuracy = 0

    def load_train_data(self, dataloader):
        """
        Loads the training data loader.

        Args:
            dataloader (torch.utils.data.DataLoader): Training data loader.
        """
        self.train_loader = dataloader

    def load_test_data(self, dataloader):
        """
        Loads the test data loader.

        Args:
            dataloader (torch.utils.data.DataLoader): Test data loader.
        """
        self.test_loader = dataloader

    def get_name(self):
        """
        Returns the name of the agent.

        Returns:
            str: Agent's name.
        """
        return self.name

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def learn(self, epochs=1, lr=0.001):
        """
        Trains the agent's neural network.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        """
        if self.train_loader is None:
            raise ValueError("Train loader not loaded. Call load_train_data first.")

        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"Agent {self.name} Epoch {epoch + 1}/{epochs}")
            self.model.nn_train(self.train_loader, optimizer)

            # Test the agent after each epoch
            test_accuracy = 0.0
            total_batches = 0
            with torch.no_grad():
                for X, y in self.test_loader:
                    test_accuracy += self.model.nn_test(X, y)
                    total_batches+=1
            if total_batches > 0:
                test_accuracy /= total_batches

            self._accuracy = test_accuracy

    def accuracy(self):
        return format_float(self._accuracy)

class NeuralNetwork(nn.Module):
    """
    A simple neural network with three fully connected layers.
    """
    def __init__(self, input_size, output_size):
        """
        Initializes the NeuralNetwork.

        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.view(x.size(0), -1) #Flatten the input.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def nn_train(self, data_loader, optimizer, epochs = 1):
        """
        Trains the neural network.

        Args:
            data_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            epochs (int, optional): Number of training epochs. Defaults to 1.
        """
        self.train()
        for _ in range(epochs):
            for batch_data, batch_labels in data_loader:
                optimizer.zero_grad()
                outputs = self(batch_data)
                loss = nn.CrossEntropyLoss()(outputs, batch_labels)
                loss.backward()
                optimizer.step()

    def nn_test(self, X, y):
        """
        Tests the neural network.

        Args:
            X (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            float: Average accuracy of the predictions.
        """
        self.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1)
            output = self.forward(X)
            predictions = torch.argmax(output, dim=1)
            return (predictions == y).float().mean().item()