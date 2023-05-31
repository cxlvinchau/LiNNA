# EXAMPLE
# This file should show how an abstraction works for one model

# imports
from torchvision import datasets, transforms
import torch
import sys
sys.path.append("..")
from linna.abstraction import Abstraction
from linna.network import Network
from linna.utils import get_accuracy
import os

# Load MNIST dataset
# Set to true to download MNIST data set
DOWNLOAD = not os.path.isdir('../datasets/MNIST/TRAINSET')

# Get the data data
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Specify the network (it should be stored as a torch-model)
network = "MNIST3x100"
print("Loading a MNIST 3x100 model")
sequential = torch.load(network, map_location=torch.device('cpu'))

# Create a Linna-Network (that contains some functionatlities)
network = Network(sequential)

# Specify the reduction rate
reduction_rate = 0.5

print("Abstract the network and reduce it by 50%")
# Specify the abstraction
abstraction = Abstraction(network=network, # which network to use (must be LiNNA-Network)
                          basis_finder="variance", # which basis finding methodd (variance or greedy)
                          coef_finder="l2", # which coefficient finding method (l1 for linear programming = gurobi, l2 for orthogonal projection)
                          loader=trainloader) # specify the dataset that should be used (=IO set)
# Generate the abstraction
abstraction.determine_bases(reduction_rate, random_choice=False)
abstraction.abstract_all()

# Test the abstraction
accuracy = get_accuracy(testloader, abstraction.network.torch_model)

print(f"Accuracy after abstraction {accuracy}, Size of the net {abstraction.network.get_num_neurons()} neurons")
