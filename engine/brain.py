import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """ Must make the number of layers dynamic """
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

    def nn_train(self, data_loader, optimizer, epochs = 1):
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
        self.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1)
            output = self.forward(X)
            predictions = torch.argmax(output, dim=1)
            return (predictions == y).float().mean().item()