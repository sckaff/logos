import torch
import torch.nn as nn
import torch.optim as optim

from utils import config
from utils.brain import NeuralNetwork
from utils.helpers import format_float

from faker import Faker

faker = Faker()

class Agent(nn.Module):
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
        self.train_loader = None
        self.test_loader = None
        self.accuracy = 0

        # Neuroevolution Variables
        self.model = NeuralNetwork(input_size, output_size)
        if name is None:
            self.name = faker.name()
        else:
            self.name = name
        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


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

    def get_accuracy(self):
        test_accuracy = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for X, y in self.test_loader:
                test_accuracy += self.model.nn_test(X, y)
                total_batches+=1
        
        if total_batches > 0:
            test_accuracy /= total_batches

        self.accuracy = test_accuracy
        return format_float(self.accuracy)
    
    def set_lr(self, lr):
        print(f"Learning rate updated to {lr}")
        self.lr = lr

    def set_optimizer(self, optimizer, lr, alpha=0, momentum=0):
        if optimizer not in config.OPTIMIZERS:
            raise ValueError("Invalid Optimizer")
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            if alpha == 0:
                raise ValueError("Please specify alpha if optimizer = RMSProp")
            self.optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.9)
        else:
            if momentum == 0:
                raise ValueError("Please specify momentum if optimizer = SGD")
            self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def learn(self, epochs=1):
        """
        Trains the agent's neural network.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        """
        if self.train_loader is None:
            raise ValueError("Train loader not loaded. Call load_train_data first.")

        self.set_optimizer("adam", self.lr)
        self.model.nn_train(self.train_loader, self.optimizer, epochs)