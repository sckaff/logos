import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch

from utils import config

# From base class to specified agent
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
# From base class to specified brain ?? Maybe.


# Evolution loop
population = [NeuralNetwork() for _ in range(population_size)]

for gen in range(generations):
    # Evaluate fitness
    fitness_scores = [evaluate(nn, X_train, y_train) for nn in population]
    
    # Select the top-k networks
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort by highest accuracy
    best_networks = [population[i] for i in sorted_indices[:top_k]]
    
    print(f"Generation {gen+1}, Best Accuracy: {fitness_scores[sorted_indices[0]]:.4f}")

    # Create new population: keep top-k and mutate
    new_population = best_networks[:]
    while len(new_population) < population_size:
        parent = np.random.choice(best_networks)  # Pick a random elite
        child = mutate(parent, mutation_rate)
        new_population.append(child)

    population = new_population  # Replace old population

# Final evaluation on test set
best_nn = best_networks[0]
test_accuracy = evaluate(best_nn, X_test, y_test)
print(f"Final Test Accuracy: {test_accuracy:.4f}")