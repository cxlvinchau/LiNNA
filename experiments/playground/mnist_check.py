from experiments.reduction.run_reduction_experiment import run_reduction_experiment, run_bisimulation
from torchvision import datasets, transforms
import torch

import numpy as np
from linna.network import Network
from linna.utils import load_tf_network

transform = transforms.Compose([transforms.ToTensor()])
# Get data
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=False, train=False, transform=transform)

# Create train and test loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Load trained neural network
sequential = load_tf_network(file="../networks/MNIST_3x100.tf")
network = Network(torch_model=sequential)

io = network.get_io_matrix(1, testloader, size=5000)

for n in range(100):
    print(n, np.sum(io[:, n]))