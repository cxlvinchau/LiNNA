import sys, os
import numpy as np
from torchvision import datasets, transforms

from experiments.playground.marabou_utils import linna_to_marabou
from linna.network import Network
from linna.utils import load_tf_network
import torch

import marabou_onnx

sys.path.append('/home/calvin/Documents/tools/Marabou')

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('../datasets/MNIST/TRAINSET', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

X, y = next(iter(trainloader))
x = X[0].view(-1, 784)[0]

from maraboupy import Marabou
from maraboupy import MarabouCore

options = Marabou.createOptions(verbosity=0)

# Export network
sequential = load_tf_network(file="../networks/MNIST_3x100.tf")
linna_net = Network(sequential)

cex = linna_to_marabou(linna_net, x=x, delta = 0)
print(cex)
print("-----------------------------")
print(linna_net.forward(x))
