from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from configs import MNIST
from engine.agent import Agent

class MNISTEnvironment:
    def __init__(self, shuffle=True, download=True, data_path='./data'):
        self.shuffle = shuffle
        self.data_path = data_path
        self.agents = []

        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        self.train_dataset = datasets.MNIST(root=self.data_path, train=True, download=download, transform=transform)
        self.test_dataset = datasets.MNIST(root=self.data_path, train=False, download=download, transform=transform)

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=MNIST.BATCH_SIZE, shuffle=self.shuffle)
        self.test_loader = DataLoader(self.test_dataset, batch_size=MNIST.BATCH_SIZE, shuffle=False) #test loader should not shuffle.
        self.train_iterator = iter(self.train_loader)
        self.test_iterator = iter(self.test_loader)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
    
    def create_agent(self, name=None):
        new_agent = Agent(MNIST.INPUT_SIZE, MNIST.OUTPUT_SIZE, name)

        self.agents.append(new_agent)

    def list_agents(self):
        for i, agent in enumerate(self.agents):
            print("Agent #" + str(i+1) + ": " + agent.get_name())    

    def train_all_agents(self, epochs=1):
        for agent in self.agents:
            print(f"\nTraining agent: {agent.get_name()}")
            agent.load_train_data(self.get_train_loader())
            agent.load_test_data(self.get_test_loader())
            agent.learn(epochs)

            print(f"Agent {agent.get_name()} Test Accuracy: {agent.get_accuracy()}")
            print("\n#-------------------#")

    def show_average_accuracy(self):
        sum_accuracy = 0
        for agent in self.agents:
            sum_accuracy += agent.accuracy()

        print(f"Average test accuracy of all agents: {sum_accuracy/len(self.agents):.4f}")
    
    def show_best_agent(self):
        best_acc = 0
        best_agent = None

        for curr_agent in self.agents:
            if curr_agent.accuracy() > best_acc:
                best_acc = curr_agent.accuracy()
                best_agent = curr_agent

        return best_agent
