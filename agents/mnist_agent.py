import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch

from utils import config
import inspect

class MnistAgent:
    def __init__(self):
        config = get_env_config()

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["subset_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)

    # Get a small subset of training data
    X_train, y_train = next(iter(train_loader))
    X_train = X_train.view(-1, 28 * 28)  # Flatten images

    X_test, y_test = next(iter(test_loader))
    X_test = X_test.view(-1, 28 * 28)

def get_env_config():
    caller_frame = inspect.currentframe().f_back  # Get the caller's frame
    caller_instance = caller_frame.f_locals.get('self', None)  # Get the caller instance
    caller_class_name = caller_instance.__class__.__name__ if caller_instance else "Unknown"

    print(f"Called by class: {caller_class_name}")  # Debugging purpose

    if caller_class_name == "MNIST_Agent":
        return {
            "population": 100,
            "generations": 50,
            "top_k": 10,
            "mutation_rate": 0.05,
            "subset_size": 5
        }
    else:
        return 0
