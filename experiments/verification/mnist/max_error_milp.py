import torch
from torchvision import datasets, transforms

from linna.network import Network
from linna.utils import load_tf_network

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(
    '../../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=False)


# Load network
sequential = load_tf_network(file=f"../../networks/MNIST_3x100.tf")
linna_net = Network(sequential)

# Layer 1
io_matrix = linna_net.get_io_matrix(layer_idx=1, loader=trainloader, size=1000)
