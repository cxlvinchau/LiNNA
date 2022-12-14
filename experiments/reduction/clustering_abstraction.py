from linna.abstraction import Abstraction
from linna.network import Network
from linna.utils import load_tf_network, get_accuracy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import pandas as pd

# Load MNIST dataset
# Set to true to download MNIST data set
DOWNLOAD = False
# Get training data
transform = transforms.Compose([transforms.ToTensor()])
# Get data
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=DOWNLOAD, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST('../datasets/MNIST/TESTSET', download=DOWNLOAD, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Load trained neural network
sequential = load_tf_network(file="../networks/MNIST_5x100.tf")
network = Network(torch_model=sequential)

rows = []

# Compute different abstractions
for basis_size in range(10, 91, 10):
    abstraction = Abstraction(network=network,
                              basis_finder="kmeans",
                              coef_finder="clustering",
                              loader=trainloader)
    for layer_idx in range(len(abstraction.network.layers) - 1):
        abstraction.determine_basis(layer_idx=layer_idx, basis_size=basis_size)
    for layer_idx in range(len(abstraction.network.layers) - 1):
        abstraction.abstract(layer_idx=layer_idx)
    print(f"Basis size: {basis_size}, Reduction rate: {abstraction.get_reduction_rate()}")
    row = pd.Series({"basis_size": basis_size,
                     "reduction_rate": abstraction.get_reduction_rate(),
                     "accuracy": get_accuracy(testloader, abstraction.network.torch_model)
                     })
    rows.append(row)
    abstraction.network.reset()

df = pd.DataFrame(rows)
df.plot.line(x="reduction_rate", y="accuracy", ylim=(0, 1))
plt.show()
