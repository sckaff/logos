from environments.mnist_env import MNISTEnvironment
env = MNISTEnvironment()

# Create multiple agents
for i in range(3):  # Create 3 agents for example
    env.create_agent()

env.train_all_agents()

# population = [BaseAgent(config.INPUT_SIZE, config.OUTPUT_SIZE) for _ in range(config.POPULATION_SIZE)]