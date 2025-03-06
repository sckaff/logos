import torch
import torch.nn as nn

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
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}")
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