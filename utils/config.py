# Neuroevolution Hyperparameters
POPULATION_SIZE = 20        # Number of agents per generation
GENERATIONS = 50            # Total number of generations
TOP_K = 5                   # Select the top-K best agents for reproduction
MUTATION_RATE = 0.05        # Strength of weight mutations

# Neural Network Settings
INPUT_SIZE = 28 * 28        # MNIST image size (flattened)
HIDDEN_SIZE = 32            # Hidden layer neurons
OUTPUT_SIZE = 10            # Output classes (digits 0-9)
ACTIVATION_FUNCTION = "tanh"  # Can be "relu", "sigmoid", "tanh", etc.

# Training Settings
SUBSET_SIZE = 1000          # Number of training samples per generation
BATCH_SIZE = 640            # Test batch size

# Logging & Debugging
LOG_INTERVAL = 10           # Print results every X generations
SAVE_RESULTS = True         # Whether to save progress to a file
