from linna.abstraction import Abstraction
from linna.network import Network
from linna.utils import load_tf_network, get_accuracy
from torchvision import datasets, transforms
import torch

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
sequential = load_tf_network(file="../networks/MNIST_3x100.tf")
network = Network(torch_model=sequential)

# Compute different abstractions
for basis_size in range(10, 91, 10):
    abstraction = Abstraction(network=network,
                              basis_finder="kmeans",
                              coef_finder="clustering",
                              loader=trainloader)
    abstraction.determine_basis(layer_idx=0, basis_size=basis_size)
    abstraction.determine_basis(layer_idx=1, basis_size=basis_size)
    abstraction.abstract(layer_idx=0)
    abstraction.abstract(layer_idx=1)
    print(f"Basis size: {basis_size}, Reduction rate: {abstraction.get_reduction_rate()}")
    print(get_accuracy(testloader, abstraction.network.torch_model))
    abstraction.network.reset()
